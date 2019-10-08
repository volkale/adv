import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from prepare_data import get_model_input_df

plt.style.use('ggplot')

parent_dir_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_lnMean_lnSD_plot():
    df = get_model_input_df(only_placebo_controled=True)
    # check whether the standard deviation of the change variable is correlated with its mean (in active and placebo)
    # and save the OLS summary as html file
    scale_list = ['HAMD17', 'HAMD21', 'HAMD24', 'HAMDunspecified', 'MADRS']
    for scale in scale_list:
        for group in [0, 1]:
            lm = sm.OLS(
                df.query(f'scale == "{scale}" & is_active == {group}')['lnSD'].values,
                sm.add_constant(df.query(f'scale == "{scale}" & is_active == {group}')['lnMean'].values)
            )
            res = lm.fit()

            with open(os.path.join(parent_dir_name, f'output/lm_res_{group}_{scale}.html'), 'w') as f:
                f.write(res.summary2().as_html())

    ax = df.query('is_active == 1').plot.scatter(
        x='lnMean', y='lnSD',
        grid=True, color='r', figsize=(8, 6), title='ln(mean of negative change) vs. ln(sd of change)'
    )
    _ = df.query('is_active == 0').plot.scatter(x='lnMean', y='lnSD', grid=True, ax=ax)  # NOQA
    _ = ax.legend(('active', 'placebo'))  # NOQA

    for sid, df_ in df.groupby('study_id'):
        df_a = df_.query('is_active == 1')
        df_p = df_.query('is_active == 0')

        plt.plot((df_a.lnMean.values[0], df_p.lnMean.values[0]),
                 (df_a.lnSD.values[0], df_p.lnSD.values[0]), c='gray', linestyle='-', alpha=0.2)

    plt.savefig(os.path.join(parent_dir_name, f'output/lnMean_lnSD_plot.png'))
    return plt
