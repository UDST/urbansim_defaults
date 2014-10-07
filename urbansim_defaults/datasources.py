import numpy as np
import pandas as pd
import os
import utils
from urbansim.utils import misc
import urbansim.sim.simulation as sim

import warnings

warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
pd.options.mode.chained_assignment = None


@sim.table_source('buildings')
def buildings(store, households, jobs, building_sqft_per_job, settings):
    df = store['buildings']

    if settings.get("set_nan_price_to_zero", False):
        for col in ["residential_sales_price", "non_residential_rent"]:
            df[col] = 0

    if settings.get("reconcile_residential_units_and_households", False):
        # prevent overfull buildings (residential)
        df["residential_units"] = pd.concat([df.residential_units,
                                             households.building_id.value_counts()
                                             ], axis=1).max(axis=1)

    if settings.get("reconcile_non_residential_sqft_and_jobs", False):
        # prevent overfull buildings (non-residential)
        tmp_df = pd.concat([
            df.non_residential_sqft,
            jobs.building_id.value_counts() *
            df.building_type_id.fillna(-1).map(building_sqft_per_job)
        ], axis=1)
        df["non_residential_sqft"] = tmp_df.max(axis=1).apply(np.ceil)

    fill_nas_cfg = settings.get("table_reprocess", {}).get("buildings", None)
    if fill_nas_cfg is not None:
        df = utils.table_reprocess(fill_nas_cfg, df)

    return df


@sim.table_source('household_controls')
def household_controls():
    df = pd.read_csv(os.path.join(misc.data_dir(), "household_controls.csv"))
    return df.set_index('year')


@sim.table_source('employment_controls')
def employment_controls():
    df = pd.read_csv(os.path.join(misc.data_dir(), "employment_controls.csv"))
    return df.set_index('year')


@sim.table_source('jobs')
def jobs(store, settings):
    df = store['jobs']

    if settings.get("remove_invalid_building_ids", True):
        # have to do it this way to prevent circular reference
        df.building_id.loc[~df.building_id.isin(store['buildings'].index)] = -1

    fill_nas_cfg = settings.get("table_reprocess", {}).get("jobs", None)
    if fill_nas_cfg is not None:
        df = utils.table_reprocess(fill_nas_cfg, df)

    return df


@sim.table_source('households')
def households(store, settings):
    df = store['households']

    if settings.get("remove_invalid_building_ids", True):
        # have to do it this way to prevent circular reference
        df.building_id.loc[~df.building_id.isin(store['buildings'].index)] = -1

    fill_nas_cfg = settings.get("table_reprocess", {}).get("households", None)
    if fill_nas_cfg is not None:
        df = utils.table_reprocess(fill_nas_cfg, df)

    return df


@sim.table_source('parcels')
def parcels(store):
    df = store['parcels']
    return df


# these are shapes - "zones" in the bay area
@sim.table_source('zones')
def zones(store):
    df = store['zones']
    return df


# these are dummy returns that last until accessibility runs
@sim.table("nodes")
def nodes():
    return pd.DataFrame()


@sim.table("logsums")
def logsums(settings):
    logsums_index = settings.get("logsums_index_col", "taz")
    return pd.read_csv(os.path.join(misc.data_dir(),
                                    'logsums.csv'),
                       index_col=logsums_index)


# this specifies the relationships between tables
sim.broadcast('nodes', 'buildings', cast_index=True, onto_on='node_id')
sim.broadcast('nodes', 'parcels', cast_index=True, onto_on='node_id')
sim.broadcast('logsums', 'buildings', cast_index=True, onto_on='zone_id')
sim.broadcast('logsums', 'parcels', cast_index=True, onto_on='zone_id')
sim.broadcast('parcels', 'buildings', cast_index=True, onto_on='parcel_id')
sim.broadcast('buildings', 'households', cast_index=True, onto_on='building_id')
sim.broadcast('buildings', 'jobs', cast_index=True, onto_on='building_id')
