from __future__ import print_function

import os
import time

import numpy as np
import orca
import pandana as pdna
import pandas as pd
from urbansim.utils import misc
from urbansim.utils import networks
from urbansim_defaults import utils

import datasources
import variables


@orca.step('rsh_estimate')
def rsh_estimate(homesales, aggregations):
    return utils.hedonic_estimate("rsh.yaml", homesales, aggregations)


@orca.step('rsh_simulate')
def rsh_simulate(buildings, aggregations):
    return utils.hedonic_simulate("rsh.yaml", buildings, aggregations,
                                  "residential_price")


@orca.step('nrh_estimate')
def nrh_estimate(costar, aggregations):
    return utils.hedonic_estimate("nrh.yaml", costar, aggregations)


@orca.step('nrh_simulate')
def nrh_simulate(buildings, aggregations):
    return utils.hedonic_simulate("nrh.yaml", buildings, aggregations,
                                  "non_residential_price")


@orca.step('hlcm_estimate')
def hlcm_estimate(households, buildings, aggregations):
    return utils.lcm_estimate("hlcm.yaml", households, "building_id",
                              buildings, aggregations)


@orca.step('hlcm_simulate')
def hlcm_simulate(households, buildings, aggregations, settings):
    return utils.lcm_simulate("hlcm.yaml", households, buildings,
                              aggregations,
                              "building_id", "residential_units",
                              "vacant_residential_units",
                              settings.get("enable_supply_correction", None))


@orca.step('elcm_estimate')
def elcm_estimate(jobs, buildings, aggregations):
    return utils.lcm_estimate("elcm.yaml", jobs, "building_id",
                              buildings, aggregations)


@orca.step('elcm_simulate')
def elcm_simulate(jobs, buildings, aggregations):
    return utils.lcm_simulate("elcm.yaml", jobs, buildings, aggregations,
                              "building_id", "job_spaces",
                              "vacant_job_spaces")


@orca.step('households_relocation')
def households_relocation(households, settings):
    rate = settings['rates']['households_relocation']
    return utils.simple_relocation(households, rate, "building_id")


@orca.step('jobs_relocation')
def jobs_relocation(jobs, settings):
    rate = settings['rates']['jobs_relocation']
    return utils.simple_relocation(jobs, rate, "building_id")


@orca.step('households_transition')
def households_transition(households, household_controls, year, settings):
    return utils.full_transition(households,
                                 household_controls,
                                 year,
                                 settings['households_transition'],
                                 "building_id")


@orca.step('simple_households_transition')
def simple_households_transition(households, settings):
    rate = settings['rates']['simple_households_transition']
    return utils.simple_transition(households, rate, "building_id")


@orca.step('jobs_transition')
def jobs_transition(jobs, employment_controls, year, settings):
    return utils.full_transition(jobs,
                                 employment_controls,
                                 year,
                                 settings['jobs_transition'],
                                 "building_id")


@orca.step('simple_jobs_transition')
def simple_jobs_transition(jobs, settings):
    rate = settings['rates']['simple_jobs_transition']
    return utils.simple_transition(jobs, rate, "building_id")


@orca.injectable('net', cache=True)
def build_networks(settings):
    name = settings['build_networks']['name']
    st = pd.HDFStore(os.path.join(misc.data_dir(), name), "r")
    nodes, edges = st.nodes, st.edges
    net = pdna.Network(nodes["x"], nodes["y"], edges["from"], edges["to"],
                       edges[["weight"]])
    net.precompute(settings['build_networks']['max_distance'])
    return net


@orca.step('neighborhood_vars')
def neighborhood_vars(net):
    nodes = networks.from_yaml(net, "neighborhood_vars.yaml")
    nodes = nodes.fillna(0)
    print(nodes.describe())
    orca.add_table("nodes", nodes)


@orca.step('price_vars')
def price_vars(net):
    nodes2 = networks.from_yaml(net, "price_vars.yaml")
    nodes2 = nodes2.fillna(0)
    print(nodes2.describe())
    nodes = orca.get_table('nodes')
    nodes = nodes.to_frame().join(nodes2)
    orca.add_table("nodes", nodes)


@orca.step('feasibility')
def feasibility(parcels, settings,
                parcel_sales_price_sqft_func,
                parcel_is_allowed_func):
    kwargs = settings['feasibility']
    utils.run_feasibility(parcels,
                          parcel_sales_price_sqft_func,
                          parcel_is_allowed_func,
                          **kwargs)


@orca.injectable("add_extra_columns_func", autocall=False)
def add_extra_columns(df):
    for col in ["residential_price", "non_residential_price"]:
        df[col] = 0
    return df


@orca.step('residential_developer')
def residential_developer(feasibility, households, buildings, parcels, year,
                          settings, summary, form_to_btype_func,
                          add_extra_columns_func):
    kwargs = settings['residential_developer']
    new_buildings = utils.run_developer(
        "residential",
        households,
        buildings,
        "residential_units",
        parcels.parcel_size,
        parcels.ave_sqft_per_unit,
        parcels.total_residential_units,
        feasibility,
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        **kwargs)

    summary.add_parcel_output(new_buildings)


@orca.step('non_residential_developer')
def non_residential_developer(feasibility, jobs, buildings, parcels, year,
                              settings, summary, form_to_btype_func,
                              add_extra_columns_func):

    kwargs = settings['non_residential_developer']
    new_buildings = utils.run_developer(
        ["office", "retail", "industrial"],
        jobs,
        buildings,
        "job_spaces",
        parcels.parcel_size,
        parcels.ave_sqft_per_unit,
        parcels.total_job_spaces,
        feasibility,
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        residential=False,
        **kwargs)

    summary.add_parcel_output(new_buildings)


@orca.step("scheduled_development_events")
def scheduled_development_events(buildings, development_projects, summary, year):
    dps = development_projects.to_frame().query("year_built == %d" % year)

    if len(dps) == 0:
        return

    new_buildings = utils.scheduled_development_events(buildings, dps,
                                       remove_developed_buildings=True,
                                       unplace_agents=['households', 'jobs'])

    summary.add_parcel_output(new_buildings)


@orca.step("diagnostic_output")
def diagnostic_output(households, buildings, parcels, zones, year, summary):
    households = households.to_frame()
    buildings = buildings.to_frame()
    parcels = parcels.to_frame()
    zones = zones.to_frame()

    zones['zoned_du'] = parcels.groupby('zone_id').zoned_du.sum()
    zones['zoned_du_underbuild'] = parcels.groupby('zone_id').\
        zoned_du_underbuild.sum()
    zones['zoned_du_underbuild_ratio'] = zones.zoned_du_underbuild /\
        zones.zoned_du

    zones['residential_units'] = buildings.groupby('zone_id').\
        residential_units.sum()
    zones['non_residential_sqft'] = buildings.groupby('zone_id').\
        non_residential_sqft.sum()

    zones['retail_sqft'] = buildings.query('general_type == "Retail"').\
        groupby('zone_id').non_residential_sqft.sum()
    zones['office_sqft'] = buildings.query('general_type == "Office"').\
        groupby('zone_id').non_residential_sqft.sum()
    zones['industrial_sqft'] = buildings.query('general_type == "Industrial"').\
        groupby('zone_id').non_residential_sqft.sum()

    zones['average_income'] = households.groupby('zone_id').income.quantile()
    zones['household_size'] = households.groupby('zone_id').persons.quantile()

    zones['building_count'] = buildings.\
        query('general_type == "Residential"').groupby('zone_id').size()
    zones['residential_price'] = buildings.\
        query('general_type == "Residential"').groupby('zone_id').\
        residential_price.quantile()
    zones['retail_rent'] = buildings[buildings.general_type == "Retail"].\
        groupby('zone_id').non_residential_price.quantile()
    zones['office_rent'] = buildings[buildings.general_type == "Office"].\
        groupby('zone_id').non_residential_price.quantile()
    zones['industrial_rent'] = \
        buildings[buildings.general_type == "Industrial"].\
        groupby('zone_id').non_residential_price.quantile()

    summary.add_zone_output(zones, "diagnostic_outputs", year)
