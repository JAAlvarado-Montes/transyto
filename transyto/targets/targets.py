#! /usr/bin/python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import datetime
import os
import requests

from matplotlib.font_manager import FontProperties
from contextlib import suppress

from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroplan import FixedTarget, Observer
from astroplan.plots import plot_airmass

from transyto.targets.swarthmore import configure_transit_finder
from transyto.utils import set_xaxis_limits


def filter_transit_observations(local_delta=10, cov_threshold=80, days_to_print=5, days_in_past=0,
                                save_tables=False, output_directory="", start_date="today",
                                min_start_elevation=30, elevation_conector="or",
                                min_end_elevation=30, min_transit_depth=5, max_magnitude=11,
                                observatory="Siding"):
    """Filter and plot transits for a given observatory using different criteria (see below).

    Parameters
    ----------
    local_delta : int, optional (default is 10)
        Delta time to calculate the local time from UTC. Ex: Sydney UTC + 10
    cov_threshold : int, optional (default is 80)
        Minimum transit coverage to show transits.
    days_to_print : int, optional (default is 5)
        Number of days from provided date to show transits.
    days_in_past : str, optional (default is 0)
        Number of days to the past of provided date.
    save_tables : bool, optional (defaul is False)
        Flag to save both the table of exoplanets and TESS candidates (Tois).
    output_directory : str, optional (Default current directory)
        Ouput directory to save tables and plots.
    start_date : str, optional  (default is "today")
        Starting date to look for transits. Either "today" or "mm-dd-yyyy" (ex. "08-20-2021")
    min_start_elevation : int, optional (default is 30)
        Minimum elevation at start of transit to look for transits.
    elevation_conector : str, optional (default is "or")
        Conditional for minimum and maximum elevation. It can be "or" and "and"
    min_end_elevation : int, optional (default is 30)
        Minimum elevation at end of transit to look for transits.
    min_transit_depth : int, optional (default is 5)
        Minimum transit depth to filter transits.
    max_magnitude : int, optional (default is 11)
        Maximum stellar magnitude to filter transits.
    observatory : str, optional (default is "Siding")
        String pattern to look for observatory.

    """

    # Configure the transit finder webpage (swarthmore) to get the transits of exoplaneta and
    # tois rom CSV tables.
    planets_file = configure_transit_finder(days_to_print=days_to_print,
                                            days_in_past=days_in_past, start_date=start_date,
                                            min_start_elevation=min_start_elevation,
                                            elevation_conector=elevation_conector,
                                            min_end_elevation=min_end_elevation,
                                            min_transit_depth=min_transit_depth,
                                            max_magnitude=max_magnitude,
                                            observatory=observatory, database="exoplanets", )

    tois_file = configure_transit_finder(days_to_print=days_to_print,
                                         days_in_past=days_in_past, start_date=start_date,
                                         min_start_elevation=min_start_elevation,
                                         elevation_conector=elevation_conector,
                                         min_end_elevation=min_end_elevation,
                                         min_transit_depth=min_transit_depth,
                                         max_magnitude=max_magnitude,
                                         observatory=observatory, database="tois")

    # Check if output directory was provided and if not then ask for path.
    if output_directory:
        output_directory = os.path.join(output_directory, "visibility_plots")

    else:
        output_directory = input("Enter the output directory: ")
        output_directory = os.path.join(output_directory, "Visibility_plots")

    # Create the output directory for tables and plots.
    os.makedirs(output_directory, exist_ok=True)

    if save_tables:
        r_exo = requests.get(planets_file, allow_redirects=True)
        r_toi = requests.get(tois_file, allow_redirects=True)

        planets_file = os.path.join(output_directory, "exoplanets.csv")
        tois_file = os.path.join(output_directory, "tois.csv")

        with open(planets_file, "wb") as file:
            file.write(r_exo.content)

        with open(tois_file, "wb") as file:
            file.write(r_toi.content)

    exop_df = pd.read_csv(planets_file, delimiter=",")
    toi_df = pd.read_csv(tois_file, delimiter=",")

    big_df = pd.concat([exop_df, toi_df], ignore_index=True, sort=False)

    # Filter by coverage >= threshold
    big_df = big_df[big_df['percent_baseline_observable'] >= cov_threshold]

    # filter by TESS magnitude <= mag_threshold
    big_df = big_df[big_df['V'] <= max_magnitude]

    # filter by transit depth >= depth_threshold
    big_df = big_df[big_df['depth(ppt)'] >= min_transit_depth]

    # Group transits by date
    big_df['new_start_date'] = pd.to_datetime(big_df['start time']).dt.date
    transit_groups = big_df.sort_values(by='percent_baseline_observable',
                                        ascending=False).groupby("new_start_date")

    dates = []
    for date, name in transit_groups:
        dates.append(date)

    parameters = {"axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12}
    plt.rcParams.update(parameters)

    for date in dates:

        fig, ax = plt.subplots(1, 1, figsize=(10.0, 8.0))

        start_dates = transit_groups.get_group(date)['new_start_date']
        start_times = transit_groups.get_group(date)['obs_start_time']
        end_times = transit_groups.get_group(date)['obs_end_time']
        target_names = transit_groups.get_group(date)['Name']
        toi_names = transit_groups.get_group(date)['TOI']
        coords = transit_groups.get_group(date)['coords(J2000)']
        perc_baselines = transit_groups.get_group(date)['percent_baseline_observable']
        depths = transit_groups.get_group(date)['depth(ppt)']
        durations = transit_groups.get_group(date)['duration(hours)']
        periods = transit_groups.get_group(date)['period(days)']
        mags = transit_groups.get_group(date)['V']

        colors = iter(plt.cm.tab20(np.linspace(0, 1, len(start_dates))))

        # Properties for the table
        data = []
        row_labels = []
        row_colors = []

        comparison_coords = []

        for (start_date, start_time, end_time, name, coord,
                perc, depth, duration, period, mag, toi) in zip(start_dates, start_times, end_times,
                                                                target_names, coords, perc_baselines,
                                                                depths, durations, periods, mags,
                                                                toi_names):
            obs_time = Time(f"{date} {start_time}")
            end_time = Time(f"{date} {end_time}")

            observatory = Observer.at_site('Siding Spring Observatory')

            if (obs_time.jd >= observatory.twilight_evening_nautical(Time(f"{start_date}")).to_value(format="jd") - 0.1
                    and end_time.jd <= observatory.twilight_morning_nautical(end_time).to_value(format="jd") + 0.1):

                with suppress(ValueError):
                    ra, dec = coord.rsplit()

                    if len(ra) == 11 and "-" not in ra:
                        comparison_1 = ra[:-4]
                    if len(ra) == 12 and "-" in ra:
                        comparison_1 = ra[:-4]
                    if len(ra) == 10 and "-" not in ra:
                        comparison_1 = ra[:-3]

                    if len(dec) == 11 and "-" not in dec:
                        comparison_2 = dec[:-4]
                    if len(dec) == 11 and "-" in dec:
                        comparison_2 = dec[:-3]
                    if len(dec) == 12 and "-" in dec:
                        comparison_2 = dec[:-4]
                    if len(dec) == 10 and "-" not in dec:
                        comparison_2 = dec[:-3]

                    if (comparison_1, comparison_2) not in comparison_coords:
                        comparison_coords.append((comparison_1, comparison_2))

                        color = next(colors)

                        obj_label = f"{name}"
                        mag_label = "V"
                        if not pd.isnull(toi):
                            obj_label = f"toi {toi}"
                            mag_label = "T"

                        data.append((perc, f"{mag_label} {mag:.1f}", duration, depth * 1000,
                                     f"{period:.3f}", ra, dec))
                        row_labels.append(obj_label)
                        row_colors.append(color)

                        target = FixedTarget(coord=SkyCoord(coord, unit=(u.hourangle, u.deg)),
                                             name=obj_label)

                        guide_style = {'color': color, 'linewidth': 1.7, "tz": "UTC"}
                        plot_airmass(target, observatory, obs_time, ax=ax, brightness_shading=True,
                                     altitude_yaxis=True, style_kwargs=guide_style)

                        # Calculate airmass at start and end time
                        airmass_start = observatory.altaz(obs_time, target).secz
                        airmass_end = observatory.altaz(end_time, target).secz
                        ax.plot_date(obs_time.plot_date, airmass_start, ls="", marker="o",
                                     ms=5.5, c=color)
                        ax.plot_date(end_time.plot_date, airmass_end, ls="", marker="o",
                                     ms=5.5, c=color)
                    else:
                        continue
            else:
                continue

            ax.set_xlabel(f"{start_date} [UTC] ➡︎", labelpad=8)
            ax.xaxis.set_label_coords(0.12, -0.11)

        colLabels = ("Coverage [%]", "mag", r"$\Delta t$ [h:m]", r"$\delta$ [ppm]",
                     r"$P_\mathrm{orb}$ [d]", "RA", "Dec")
        col_colors = plt.cm.BuPu(np.full(len(colLabels), 0.1))

        the_table = ax.table(cellText=data, rowLoc='right', rowColours=row_colors,
                             colColours=col_colors, rowLabels=row_labels, colLabels=colLabels,
                             colLoc='center', loc='bottom', bbox=[0, -0.6, 1, 0.4],
                             cellLoc='center')

        cells = the_table.get_celld()
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(6)
        # the_table.set_in_layout(True)
        # the_table.scale(5, 10)
        for (row, col), cell in cells.items():
            # cell.padding = 100
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))

        morning_twilight = observatory.twilight_morning_nautical(obs_time).plot_date

        evening_twilight = observatory.twilight_evening_nautical(obs_time).plot_date

        observatory_midnight = observatory.midnight(obs_time).plot_date

        # For the minor ticks, use no labels; default NullFormatter.
        ax.tick_params(axis="x", which="major", length=7)
        ax.tick_params(axis="x", which="minor", length=3)
        ax.tick_params(axis="both", which="both", direction="in")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='center')

        ax.xaxis.set_minor_locator(plticker.AutoMinorLocator())
        # ax.xaxis.set_minor_locator(dates.MinuteLocator(interval=45))

        ax.axvline(morning_twilight, c="k", ls="--", lw=0.8, zorder=1)
        ax.axvline(evening_twilight, c="k", ls="--", lw=0.8, zorder=1)
        ax.axvline(observatory_midnight, c="firebrick", ls="--", lw=1.5, zorder=1)
        ax.annotate("MT", (morning_twilight, 1.07), fontsize=9, ha="center", va="top",
                    bbox={'facecolor': 'wheat', 'alpha': 1.0, 'pad': 0.15, 'boxstyle': 'round'})
        ax.annotate("ET", (evening_twilight, 1.07), fontsize=9, ha="center", va="top",
                    bbox={'facecolor': 'wheat', 'alpha': 1.0, 'pad': 0.15, 'boxstyle': 'round'})
        # ax.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.17))
        ax.grid(alpha=0.6, which="both", color="w")

        ax1 = ax.twiny()
        ax1.plot_date(obs_time.plot_date, airmass_start, ls="", marker="o", ms=5.5,
                      c=color, tz="AEST")

        ticks = set_xaxis_limits(ax, ax1)
        ax1.xaxis.set_major_locator(plticker.FixedLocator(ticks))
        ax1.xaxis.set_minor_locator(plticker.AutoMinorLocator())

        delta_time = datetime.timedelta(hours=local_delta)

        local_times = [datetime.datetime.strptime(item.get_text(), "%H:%M") + delta_time
                       for item in ax.get_xticklabels()]

        local_labels = [local.strftime("%H:%M") for local in local_times]
        ax1.set_xticklabels(local_labels)

        ax1.set_xlabel(f"{start_date} [LT] ➡︎")
        ax1.xaxis.set_label_coords(0.11, 1.13)

        new_date = start_date + datetime.timedelta(days=1)
        trans = ax.get_xaxis_transform()  # x in data untis, y in axes fraction
        ax.annotate(f"{new_date} [LT] ➡︎", (observatory_midnight, 1.13), fontsize=14,
                    ha="left", va="center", xycoords=trans, c="firebrick")
        ax.annotate(f"{new_date} [UTC] ➡︎", (observatory_midnight, -0.14), fontsize=14,
                    ha="left", va="center", xycoords=trans, c="firebrick")

        ax1.tick_params(axis="x", which="major", length=7, rotation=20)
        ax1.tick_params(axis="x", which="minor", length=3)
        ax1.tick_params(axis="both", which="both", direction="in")

        fig.tight_layout(rect=[0., -0.1, 1., 1.])  # , pad=0.4, w_pad=0.5, h_pad=1.0)
        fig_name = os.path.join(output_directory, f"filtered_{start_date}.png")
        fig.savefig(fig_name, facecolor="w", dpi=300)
        plt.close(fig)

        print(f"Finished with transits on {start_date}")
