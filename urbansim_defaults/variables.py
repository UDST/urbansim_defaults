import numpy as np
import pandas as pd
from urbansim.utils import misc
import urbansim.sim.simulation as sim
import datasources
import utils


#####################
# BUILDINGS VARIABLES
#####################


@sim.column('buildings', 'node_id', cache=True)
def node_id(buildings, parcels):
    return misc.reindex(parcels.node_id, buildings.parcel_id)


@sim.column('buildings', 'zone_id', cache=True)
def zone_id(buildings, parcels):
    return misc.reindex(parcels.zone_id, buildings.parcel_id)


@sim.column('buildings', 'general_type', cache=True)
def general_type(buildings, building_type_map):
    return buildings.building_type_id.map(building_type_map)


@sim.column('buildings', 'sqft_per_unit', cache=True)
def unit_sqft(buildings):
    return buildings.building_sqft / buildings.residential_units.replace(0, 1)


@sim.column('buildings', 'lot_size_per_unit', cache=True)
def lot_size_per_unit(buildings, parcels):
    return misc.reindex(parcels.lot_size_per_unit, buildings.parcel_id)


@sim.column('buildings', 'sqft_per_job', cache=True)
def sqft_per_job(buildings, building_sqft_per_job):
    return buildings.building_type_id.fillna(-1).map(building_sqft_per_job)


@sim.column('buildings', 'job_spaces', cache=True)
def job_spaces(buildings):
    return (buildings.non_residential_sqft /
            buildings.sqft_per_job).fillna(0).astype('int')


@sim.column('buildings', 'vacant_residential_units')
def vacant_residential_units(buildings, households):
    return buildings.residential_units.sub(
        households.building_id.value_counts(), fill_value=0)


@sim.column('buildings', 'vacant_job_spaces')
def vacant_job_spaces(buildings, jobs):
    return buildings.job_spaces.sub(
        jobs.building_id.value_counts(), fill_value=0)


#####################
# HOUSEHOLDS VARIABLES
#####################


@sim.column('households', 'income_quartile', cache=True)
def income_quartile(households):
    s = pd.Series(pd.qcut(households.income, 4, labels=False),
                  index=households.index)
    # convert income quartile from 0-3 to 1-4
    s = s.add(1)
    return s

@sim.column('households', 'zone_id', cache=True)
def zone_id(households, buildings):
    return misc.reindex(buildings.zone_id, households.building_id)


@sim.column('households', 'node_id', cache=True)
def node_id(households, buildings):
    return misc.reindex(buildings.node_id, households.building_id)


#####################
# JOBS VARIABLES
#####################


@sim.column('jobs', 'node_id', cache=True)
def node_id(jobs, buildings):
    return misc.reindex(buildings.node_id, jobs.building_id)


@sim.column('jobs', 'zone_id', cache=True)
def zone_id(jobs, buildings):
    return misc.reindex(buildings.zone_id, jobs.building_id)


#####################
# PARCELS VARIABLES
#####################


@sim.column('parcels', 'parcel_size', cache=True)
def parcel_size(parcels, settings):
    return parcels.shape_area * settings.get('parcel_size_factor', 1)


@sim.column('parcels', 'parcel_acres', cache=True)
def parcel_acres(parcels):
    # parcel_size needs to be in sqft
    return parcels.parcel_size / 43560.0

@sim.column('parcels', 'total_residential_units', cache=False)
def total_residential_units(parcels, buildings):
    return buildings.residential_units.groupby(buildings.parcel_id).sum().\
        reindex(parcels.index).fillna(0)


@sim.column('parcels', 'total_job_spaces', cache=False)
def total_job_spaces(parcels, buildings):
    return buildings.job_spaces.groupby(buildings.parcel_id).sum().\
        reindex(parcels.index).fillna(0)


@sim.column('parcels', 'total_sqft', cache=True)
def total_sqft(parcels, buildings):
    return buildings.building_sqft.groupby(buildings.parcel_id).sum().\
        reindex(parcels.index).fillna(0)


@sim.column('parcels', 'zoned_du', cache=True)
def zoned_du(parcels):
    return (parcels.max_dua * parcels.parcel_acres).\
        reindex(parcels.index).fillna(0).round().astype('int')


@sim.column('parcels', 'zoned_du_underbuild')
def zoned_du_underbuild(parcels):
    return (parcels.zoned_du - parcels.total_residential_units).clip(lower=0)


@sim.column('parcels', 'ave_sqft_per_unit')
def ave_sqft_per_unit(parcels, nodes, settings):
    if len(nodes) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    s = misc.reindex(nodes.ave_sqft_per_unit, parcels.node_id)
    clip = settings.get("ave_sqft_per_unit_clip", None)
    if clip is not None:
        s = s.clip(lower=clip['lower'], upper=clip['upper'])
    return s


# this just changes the column name for reverse compatibility
@sim.column('parcels', 'ave_unit_size')
def ave_unit_size(parcels):
    return parcels.ave_sqft_per_unit


@sim.column('parcels', 'lot_size_per_unit')
def log_size_per_unit(parcels):
    return parcels.parcel_size / parcels.total_residential_units.replace(0, 1)


# returns the oldest building on the land and fills missing values with 9999 -
# for use with historical preservation
@sim.column('parcels', 'oldest_building')
def oldest_building(parcels, buildings):
    return buildings.year_built.groupby(buildings.parcel_id).min().\
        reindex(parcels.index).fillna(9999)
