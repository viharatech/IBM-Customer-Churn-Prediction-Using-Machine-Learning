import matplotlib.pyplot as plt
from logger import Logger
log = Logger.get_logs('visual')
import matplotlib.pyplot as plt
from scipy import stats

def plot_kde_comparison(df, transform_suffix, color, label):
    for col in df.columns:
        if col.endswith(transform_suffix):
            plt.figure(figsize=(10, 4))

            # KDE and boxplot in one figure (2 subplots)
            plt.subplot(1, 3, 1)
            df[col].plot(kind='kde', color=color, label=label)
            plt.title(f'{label} - {col}')
            plt.legend()

            plt.subplot(1, 3, 2)
            plt.boxplot(df[col].dropna(), labels=[label])
            plt.title(f'Boxplot - {col}')

            plt.subplot(1, 3, 3)
            stats.probplot(df[col].dropna(), dist='norm', plot=plt)
            plt.title('Probplot')

            plt.tight_layout()
            plt.savefig(f'./image/{label}-{transform_suffix}-{col}.jpeg')
            plt.show()
            log.info(f'{col} graph {label} completed')