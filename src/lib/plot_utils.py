import arviz as az


def display_hpd(ax, values, credible_interval=0.95):
    hpd_intervals = az.stats.hpd(values, credible_interval=credible_interval)

    def format_as_percent(x, round_to=0):
        return "{0:.{1:d}f}%".format(100 * x, round_to)

    plot_height = ax.get_ylim()[1]

    ax.plot(
        hpd_intervals,
        (plot_height * 0.02, plot_height * 0.02),
        lw=2,
        color="k",
        solid_capstyle="round",
    )
    ax.text(
        hpd_intervals[0],
        plot_height * 0.07,
        '{number:.{digits}f}'.format(number=hpd_intervals[0], digits=3),
        horizontalalignment="center",
    )
    ax.text(
        hpd_intervals[1],
        plot_height * 0.07,
        '{number:.{digits}f}'.format(number=hpd_intervals[1], digits=3),
        horizontalalignment="center",
    )
    ax.text(
        (hpd_intervals[0] + hpd_intervals[1]) / 2,
        plot_height * 0.3,
        format_as_percent(credible_interval) + " HPD",
        horizontalalignment="center",
    )
