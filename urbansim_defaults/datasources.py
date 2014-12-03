import numpy as np
import pandas as pd
import os
import utils
import uuid
import yaml
from urbansim.utils import misc
import urbansim.sim.simulation as sim

import warnings

warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
pd.options.mode.chained_assignment = None


@sim.injectable('settings', cache=True)
def settings():
    with open(os.path.join(misc.configs_dir(), "settings.yaml")) as f:
        settings = yaml.load(f)
        # monkey patch on the settings object since it's pretty global
        # but will also be available as injectable
        sim.settings = settings
        return settings


@sim.injectable('run_number')
def run_number():
    return misc.get_run_number()


@sim.injectable('uuid', cache=True)
def uuid_hex():
    return uuid.uuid4().hex


@sim.injectable('store', cache=True)
def hdfstore(settings):
    return pd.HDFStore(
        os.path.join(misc.data_dir(), settings["store"]),
        mode='r')


@sim.injectable("summary", cache=True)
def simulation_summary_data(run_number):
    return utils.SimulationSummaryData(run_number)


@sim.injectable("building_type_map")
def building_type_map(settings):
    return settings["building_type_map"]


@sim.injectable("scenario")
def scenario(settings):
    return settings["scenario"]


@sim.injectable("scenario_inputs")
def scenario_inputs(settings):
    return settings["scenario_inputs"]


@sim.injectable("aggregations")
def aggregations(settings):
    if "aggregation_tables" not in settings or \
    	settings["aggregation_tables"] is None:
    	return []
    return [sim.get_table(tbl) for tbl in settings["aggregation_tables"]]


@sim.injectable('building_sqft_per_job', cache=True)
def building_sqft_per_job(settings):
    return settings['building_sqft_per_job']


@sim.table('buildings', cache=True)
def buildings(store, households, jobs, building_sqft_per_job, settings):
    df = store['buildings']

    if settings.get("set_nan_price_to_zero", False):
        for col in ["residential_price", "non_residential_price"]:
            df[col] = 0

    if settings.get("reconcile_residential_units_and_households", False):
        # prevent overfull buildings (residential)
        df["residential_units"] = pd.concat(
            [df.residential_units, households.building_id.value_counts()],
            axis=1).max(axis=1)

    if settings.get("reconcile_non_residential_sqft_and_jobs", False):
        # prevent overfull buildings (non-residential)
        tmp_df = pd.concat([
            df.non_residential_sqft,
            jobs.building_id.value_counts() *
            df.building_type_id.fillna(-1).map(building_sqft_per_job)
        ], axis=1)
        df["non_residential_sqft"] = tmp_df.max(axis=1).apply(np.ceil)

    fill_nas_cfg = settings.get("table_reprocess", None)
    if fill_nas_cfg is not None:
        fill_nas_cfg = fill_nas_cfg.get("buildings", None)
    if fill_nas_cfg is not None:
        df = utils.table_reprocess(fill_nas_cfg, df)


    return df


@sim.table('household_controls', cache=True)
def household_controls():
    df = pd.read_csv(os.path.join(misc.data_dir(), "household_controls.csv"))
    return df.set_index('year')


@sim.table('employment_controls', cache=True)
def employment_controls():
    df = pd.read_csv(os.path.join(misc.data_dir(), "employment_controls.csv"))
    return df.set_index('year')


@sim.table('jobs', cache=True)
def jobs(store, settings):
    df = store['jobs']

    if settings.get("remove_invalid_building_ids", True):
        # have to do it this way to prevent circular reference
        df.building_id.loc[~df.building_id.isin(store['buildings'].index)] = -1

    fill_nas_cfg = settings.get("table_reprocess", {}).get("jobs", None)
    if fill_nas_cfg is not None:
        df = utils.table_reprocess(fill_nas_cfg, df)

    return df


@sim.table('households', cache=True)
def households(store, settings):
    df = store['households']

    if settings.get("remove_invalid_building_ids", True):
        # have to do it this way to prevent circular reference
        df.building_id.loc[~df.building_id.isin(store['buildings'].index)] = -1

    fill_nas_cfg = settings.get("table_reprocess", None)
    if fill_nas_cfg is not None:
	    fill_nas_cfg = fill_nas_cfg.get("households", None)
    if fill_nas_cfg is not None:
        df = utils.table_reprocess(fill_nas_cfg, df)

    return df


@sim.table('parcels', cache=True)
def parcels(store):
    df = store['parcels']
    return df


# these are shapes - "zones" in the bay area
@sim.table('zones', cache=True)
def zones(store):
    df = store['zones']
    return df


# these are dummy returns that last until accessibility runs
@sim.table("nodes", cache=True)
def nodes():
    return pd.DataFrame()


@sim.table("logsums", cache=True)
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
sim.broadcast(
    'buildings', 'households', cast_index=True, onto_on='building_id')
sim.broadcast('buildings', 'jobs', cast_index=True, onto_on='building_id')
