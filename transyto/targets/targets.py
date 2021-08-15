#! /usr/bin/python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import datetime
import os

from matplotlib.font_manager import FontProperties
from contextlib import suppress

from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroplan import FixedTarget, Observer
from astroplan.plots import plot_airmass

from transyto.utils import set_xaxis_limits


def filter_transit_observations(exoplanet_file="", toi_file="", local_delta=10,
                                cov_threshold=80, mag_threshold=11,
                                depth_threshold=5):
    exop_df = pd.read_csv(exoplanet_file, delimiter=",")
    toi_df = pd.read_csv(toi_file, delimiter=",")

    big_df = pd.concat([exop_df, toi_df], ignore_index=True, sort=False)

    # Filter by coverage >= threshold
    big_df = big_df[big_df['percent_baseline_observable'] >= cov_threshold]

    # filter by TESS magnitude <= mag_threshold
    big_df = big_df[big_df['V'] <= mag_threshold]

    # filter by transit depth >= depth_threshold
    big_df = big_df[big_df['depth(ppt)'] >= depth_threshold]

    # Group transits by date
    big_df['new_start_date'] = pd.to_datetime(big_df['start time']).dt.date
    transit_groups = big_df.sort_values(by='percent_baseline_observable',
                                        ascending=False).groupby("new_start_date")

    output_directory = os.path.join(os.path.dirname(exoplanet_file), "visibility_plots")
    os.makedirs(output_directory, exist_ok=True)

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

        for (start_date, start_time, end_time, name, coord,
                perc, depth, duration, period, mag, toi) in zip(start_dates, start_times, end_times,
                                                                target_names, coords, perc_baselines,
                                                                depths, durations, periods, mags,
                                                                toi_names):
            obs_time = Time(f"{date} {start_time}")
            end_time = Time(f"{date} {end_time}")

            observatory = Observer.at_site('Siding Spring Observatory')

            color = next(colors)

            if (obs_time.jd >= observatory.twilight_evening_nautical(Time(f"{start_date}")).to_value(format="jd") - 0.1
                    and end_time.jd <= observatory.twilight_morning_nautical(end_time).to_value(format="jd") + 0.1):

                with suppress(ValueError):

                    obj_label = f"{name}"
                    mag_label = "V"
                    if not pd.isnull(toi):
                        obj_label = f"toi {toi}"
                        mag_label = "T"

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

                    ra, dec = coord.rsplit()

                    data.append((ra, dec, f"{mag_label} {mag:.1f}", duration,
                                 depth * 1000, perc, f"{period:.3f}"))
                    row_labels.append(obj_label)
                    row_colors.append(color)

            else:
                continue

            ax.set_xlabel(f"{start_date} [UTC]", labelpad=8)

        colLabels = ("RA", "Dec", "mag", r"$\Delta t$ [h:m]", r"$\delta$ [ppm]",
                     "Coverage [%]", r"$P_\mathrm{orb}$ [d]")
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
        ax.axvline(observatory_midnight, c="firebrick", ls="--", lw=1.2, zorder=1)
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
        ax1.xaxis.set_label_coords(0.11, 1.15)

        new_date = start_date + datetime.timedelta(days=1)
        trans = ax.get_xaxis_transform()  # x in data untis, y in axes fraction
        ax.annotate(f"{new_date} [LT] ➡︎", (observatory_midnight, 1.15), fontsize=14,
                    ha="left", va="center", xycoords=trans)

        ax1.tick_params(axis="x", which="major", length=7, rotation=20)
        ax1.tick_params(axis="x", which="minor", length=3)
        ax1.tick_params(axis="both", which="both", direction="in")

        fig.tight_layout(rect=[0., -0.1, 1., 1.])  # , pad=0.4, w_pad=0.5, h_pad=1.0)
        fig_name = os.path.join(output_directory, f"filtered_{start_date}.png")
        fig.savefig(fig_name, facecolor="w", dpi=300)
        plt.close(fig)

        print(f"Finished with transits on {start_date}")
