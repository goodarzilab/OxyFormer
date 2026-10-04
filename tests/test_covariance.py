"""Hand-calculated, ID-aligned covariance references on synthetic units only."""
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from oxyformer.estimation.covariance import (
    AlignedInfluence, EARTH_RADIUS_KM, align_estimates, cluster_covariance,
    earth_centered_coordinates, spatial_covariance, spatial_kernel,
    spatial_sensitivities, survey_covariance,
)
from oxyformer.estimation.mtp import one_step
from oxyformer.provenance import ContractError
from test_scores import make_fixture


def influence():
    return AlignedInfluence(("a","b","c","d","e","f"),("y","z"),
                            ((1,2),(-.5,1),(.5,-1),(2,1),(-1,-2),(-2,-1)))


def groups():
    return dict(zip(influence().original_ids,("A","A","B","B","C","C")))


def locations():
    return {"A":(0.,0.), "B":(0.,1.), "C":(.5,2.)}


def test_cluster_full_covariance_independent_reference():
    # Group sums: A=(.5,3), B=(2.5,0), C=(-3,-3).
    expected = 1.5*(np.outer([.5,3],[.5,3])+np.outer([2.5,0],[2.5,0])+np.outer([-3,-3],[-3,-3]))
    result = cluster_covariance(influence(),groups(),interpretation="geographic_process")
    assert_allclose(result.matrix, expected)
    assert result.matrix[0][1] != 0
    assert_allclose(result.standard_errors,np.sqrt(np.diag(expected)))
    assert result.dependence_units == 3
    assert min(np.linalg.eigvalsh(result.matrix)) >= -1e-12


def test_align_estimates_by_original_id_before_cross_endpoint_products():
    p,data,split,n,w = make_fixture()
    first = one_step(n,data,w,n.spec,split=split,policy=p)
    # Second endpoint is exactly -2 times the first, in reverse row order.
    second = replace(first,spec=replace(first.spec,endpoint="second",outcome_scale="other units"),
                     original_ids=first.original_ids[::-1], scores=tuple(-2*x for x in first.scores[::-1]),
                     influence=tuple(-2*x for x in first.influence[::-1]),value=-2*first.value)
    aligned = align_estimates({"first":first,"second":second})
    assert aligned.original_ids == first.original_ids
    assert_allclose(np.asarray(aligned.values)[:,1],-2*np.asarray(first.influence))
    g = dict(zip(first.original_ids,("x","x","y","z")))
    v = np.asarray(cluster_covariance(aligned,g,interpretation="geographic_process").matrix)
    assert_allclose(v,v[0,0]*np.array([[1,-2],[-2,4]]))
    with pytest.raises(ContractError,match="endpoint original IDs mismatch"):
        wrong = replace(second,original_ids=("wrong",)+second.original_ids[1:],
                        lineage=replace(second.lineage,unit_ids=("wrong",)+second.original_ids[1:]))
        align_estimates({"first":first,"second":wrong})


def test_duplicating_cluster_members_does_not_create_precision():
    original = influence()
    # Duplicate every original record k times with its weight split k ways.
    # Distinct IDs represent copied records, but each remains in its old cluster.
    k = 5
    ids = tuple(f"{oid}-{j}" for oid in original.original_ids for j in range(k))
    values = tuple(tuple(np.asarray(row)/k) for row in original.values for _ in range(k))
    duplicate = AlignedInfluence(ids, original.endpoints, values)
    g = {f"{oid}-{j}":groups()[oid] for oid in original.original_ids for j in range(k)}
    a = cluster_covariance(original,groups(),interpretation="geographic_process")
    b = cluster_covariance(duplicate,g,interpretation="geographic_process")
    assert_allclose(a.matrix,b.matrix,atol=1e-13)
    assert a.dependence_units == b.dependence_units == 3
    va = spatial_covariance(original,groups(),locations(),100,interpretation="geographic_process")
    vb = spatial_covariance(duplicate,g,locations(),100,interpretation="geographic_process")
    assert_allclose(va.matrix,vb.matrix,atol=1e-13)


@pytest.mark.parametrize("method",["cluster","spatial","survey"])
def test_one_dependence_unit_is_rejected(method):
    g = dict.fromkeys(influence().original_ids,"only")
    with pytest.raises(ContractError,match="at least two"):
        if method == "cluster":
            cluster_covariance(influence(),g,interpretation="geographic_process")
        elif method == "spatial":
            spatial_covariance(influence(),g,{"only":(0,0)},100,interpretation="geographic_process")
        else:
            survey_covariance(influence(),g,g)


def test_chord_distance_convention_and_antimeridian():
    xyz = earth_centered_coordinates([(0,0),(0,90),(90,0)])
    assert_allclose(xyz,EARTH_RADIUS_KM*np.eye(3),atol=1e-12)
    # Quarter-circumference chord is sqrt(2) R; the geodesic arc is pi R/2.
    k = spatial_kernel([(0,0),(0,90)],EARTH_RADIUS_KM)
    assert k[0,1] == pytest.approx(np.exp(-1))
    assert k[0,1] != pytest.approx(np.exp(-.5*(np.pi/2)**2))
    near = spatial_kernel([(10,179.9),(10,-179.9),(90,-180),(90,180)],50)
    assert near[0,1] > .9
    assert near[2,3] == pytest.approx(1.)


@pytest.mark.parametrize("bandwidth",[50.,100.,200.])
def test_spatial_kernel_and_full_covariance_psd_and_reference(bandwidth):
    loc = locations()
    labels = ("A","B","C")
    totals = np.array([[.5,3],[2.5,0],[-3,-3]])
    # Independent spherical chord formula from great-circle central angles.
    coordinates = np.deg2rad([loc[label] for label in labels])
    kernel = np.empty((3,3))
    for i,(lat,lon) in enumerate(coordinates):
        for j,(other_lat,other_lon) in enumerate(coordinates):
            haversine = np.sin((lat-other_lat)/2)**2 + np.cos(lat)*np.cos(other_lat)*np.sin((lon-other_lon)/2)**2
            chord2 = 4*EARTH_RADIUS_KM**2*haversine
            kernel[i,j] = np.exp(-chord2/(2*bandwidth**2))
    expected = sum(kernel[i,j]*np.outer(totals[i],totals[j]) for i in range(3) for j in range(3))
    result = spatial_covariance(influence(),groups(),loc,bandwidth,interpretation="geographic_process")
    assert_allclose(result.matrix,expected,rtol=2e-14,atol=1e-13)
    assert min(np.linalg.eigvalsh(result.matrix)) >= -1e-12
    corrected = spatial_covariance(influence(),groups(),loc,bandwidth,
                                   interpretation="geographic_process",finite_cluster_correction=True)
    assert_allclose(corrected.matrix,1.5*np.asarray(result.matrix))
    rng = np.random.default_rng(104)
    for points in [np.column_stack((rng.uniform(-90,90,100),rng.uniform(-180,180,100))),
                   np.column_stack((rng.normal(0,.2,100),rng.normal(0,.2,100)))]:
        k = spatial_kernel(points,bandwidth)
        assert_allclose(k,k.T)
        assert min(np.linalg.eigvalsh(k)) >= -1e-11


def test_spatial_sensitivities_and_fixed_frame_rejection():
    results = spatial_sensitivities(influence(),groups(),locations(),interpretation="geographic_process")
    assert tuple(results) == (50,100,200)
    for b,result in results.items():
        assert float(dict(result.details)["bandwidth_km"]) == b
        assert "chord" in dict(result.details)["distance"]
    with pytest.raises(ContractError,match="separate contract"):
        cluster_covariance(influence(),groups(),interpretation="fixed_frame")
    with pytest.raises(ContractError,match="geographic_process"):
        spatial_covariance(influence(),groups(),locations(),50,interpretation="fixed_frame")


def survey_design():
    ids = influence().original_ids
    # Reused PSU labels belong to distinct PSUs across strata.
    return dict(zip(ids,("s1","s1","s1","s2","s2","s2"))), dict(zip(ids,("p1","p1","p2","p1","p2","p3")))


def test_stratified_psu_full_covariance_and_optional_fpc():
    strata,psu = survey_design()
    # Stratum 1: (.5,3), (.5,-1), centered (0,+/-2).
    first = np.array([[0,0],[0,16]])
    # Stratum 2: (2,1), (-1,-2), (-2,-1).
    u = np.array([[2,1],[-1,-2],[-2,-1]],dtype=float)
    centered = u-u.mean(axis=0)
    second = 1.5*sum(np.outer(row,row) for row in centered)
    result = survey_covariance(influence(),strata,psu)
    assert_allclose(result.matrix,first+second)
    assert result.dependence_units == 5
    assert result.interpretation == "survey_design"
    corrected = survey_covariance(influence(),strata,psu,finite_population_fraction={"s1":.25,"s2":.5})
    assert_allclose(corrected.matrix,.75*first+.5*second)
    assert min(np.linalg.eigvalsh(corrected.matrix)) >= -1e-12


def test_singleton_strata_rejected_or_explicitly_declared_certainty():
    strata,psu = survey_design()
    psu.update(a="p1",b="p1",c="p1")
    with pytest.raises(ContractError,match="singleton stratum: s1"):
        survey_covariance(influence(),strata,psu)
    result = survey_covariance(influence(),strata,psu,singleton="certainty")
    u = np.array([[2,1],[-1,-2],[-2,-1]],dtype=float)
    centered = u-u.mean(axis=0)
    assert_allclose(result.matrix,1.5*centered.T@centered)
    assert "s1" in dict(result.details)["certainty_strata"]
    with pytest.raises(ContractError,match="certainty singleton requires"):
        survey_covariance(influence(),strata,psu,singleton="certainty",finite_population_fraction={"s1":0,"s2":0})
    certified = survey_covariance(influence(),strata,psu,singleton="certainty",finite_population_fraction={"s1":1,"s2":0})
    assert_allclose(certified.matrix,result.matrix)


def test_duplication_inside_psus_preserves_survey_variance():
    original = influence()
    strata,psu = survey_design()
    ids = tuple(f"{oid}-{j}" for oid in original.original_ids for j in range(2))
    duplicate = AlignedInfluence(ids,original.endpoints,tuple(tuple(np.array(row)/2) for row in original.values for _ in range(2)))
    ss = {f"{oid}-{j}":strata[oid] for oid in original.original_ids for j in range(2)}
    pp = {f"{oid}-{j}":psu[oid] for oid in original.original_ids for j in range(2)}
    assert_allclose(survey_covariance(original,strata,psu).matrix,survey_covariance(duplicate,ss,pp).matrix)


def test_alignment_missing_groups_locations_and_invalid_design_are_rejected():
    with pytest.raises(ContractError,match="unique original"):
        AlignedInfluence(("a","a"),("x",),((1,),(-1,)))
    with pytest.raises(ContractError,match="cluster observation IDs mismatch"):
        cluster_covariance(influence(),{"a":"A"},interpretation="geographic_process")
    with pytest.raises(ContractError,match="location IDs mismatch"):
        spatial_covariance(influence(),groups(),{"A":(0,0)},50,interpretation="geographic_process")
    with pytest.raises(ContractError,match="degree bounds"):
        spatial_kernel([(91,0)],100)
    with pytest.raises(ContractError,match="bandwidth"):
        spatial_kernel([(0,0)],0)
    strata,psu = survey_design()
    with pytest.raises(ContractError,match="sampling fraction"):
        survey_covariance(influence(),strata,psu,finite_population_fraction={"s1":-1,"s2":0})
