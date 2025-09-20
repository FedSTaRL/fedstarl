from typing import List, Dict, Literal, Tuple
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from .data_object import SequenceData, SequenceBatchData

from matplotlib.ticker import AutoMinorLocator, MultipleLocator


__all__ = []


import matplotlib.pyplot as plt
import numpy as np

# Function to format numbers with 'k' suffix
def format_k(num):
    if num // 1000 != 0:
        return f'{int(num // 1000)}k'
    else:
        return f'{int(num // 1)}'

# Function to calculate dynamic font size based on bar height
def dynamic_font_size(height, max_size=10):
    return min(max_size, max(8, height / 100000))
    

def create_bar_plot(
        ds: Dict[str, SequenceData], 
        num_categories: int = 2, 
        linthresh: int = 10000, 
        alignment: Literal['vertical', 'v', 'horizontal', 'h']='vertical', 
        figsize: Tuple[int, int]=(20, 16),
        font_properties = {
            'family': 'times new roman',
            'size': 12,
            'style': 'normal',
            'weight': 'normal',
            #'variant': 'small-caps'
        },
        font_properties_title = {
            'family': 'times new roman',
            'size': 14,
            'style': 'normal',
            'weight': 'normal',
            #'variant': 'small-caps'
        },
        title: str=None,
        axis_major: int=10,
        axis_minor: int=5
    ):
    """ 
    Plot bar plot of the client distributions.
    """
    labels = np.array([ds._idx_to_user_id[k] for k in ds._user_id_to_data.keys()])
    # Number of categories to stack
    values = np.zeros((len(labels), num_categories))

    for i, ds_data in enumerate(ds._user_id_to_data.values()):
        u, c = np.unique(ds_data.labels.numpy(), return_counts=True)
        values[i, :] = c

    # Create the bar plot
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Colors for each category
    #colors = ['blue', 'orange'] #plt.cm.viridis(np.linspace(0, 1, num_categories))
    colors = ['dodgerblue', 'darkorange']
    title = f'{ds.__class__.__name__}(mode={ds._mode})' if title is None else title
    if alignment in ['vertical', 'v']:
        _bar(ax, num_categories, labels, values, colors, linthresh, u, title,
            font_properties=font_properties, font_properties_title=font_properties_title, 
            axis_major=axis_major, axis_minor=axis_minor)
    elif alignment in ['h', 'h']:
        _barh(ax, num_categories, labels, values, colors, linthresh, u, title,
            font_properties=font_properties, font_properties_title=font_properties_title, 
            axis_major=axis_major, axis_minor=axis_minor
            )
    else:
        raise ValueError(f"{alignment} not supported. Try either ['vertical', 'v', 'horizontal', 'h']")
        

def _bar(
        ax, 
        num_categories, 
        labels, 
        values, 
        colors, 
        linthresh, 
        u, 
        title: str='',
        font_properties = {
            'family': 'times new roman',
            'size': 14,
            'style': 'normal',
            'weight': 'normal',
            #'variant': 'small-caps'
        },
        font_properties_title = {
            'family': 'times new roman',
            'size': 16,
            'style': 'normal',
            'weight': 'normal',
            #'variant': 'small-caps'
        },
        axis_major: int=10,
        axis_minor: int=5):
    # Plot each category
    for i in range(num_categories):
        if i == 0:
            bars = ax.bar(labels, values[:, i], color=colors[i], label=f'{u[i]}')
        else:
            bars = ax.bar(labels, values[:, i], bottom=np.sum(values[:, :i], axis=1), color=colors[i], label=f'{u[i]}')

        # Add text annotations for each bar
        for bar in bars:
            height = bar.get_height()
            fontsize = dynamic_font_size(height)
            ax.annotate(f'{format_k(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2),
                        xytext=(0, 0),  # Offset text position
                        textcoords="offset points",
                        ha='center', va='center', color='white', fontsize=8, rotation=90)

    # Add total annotations above the bars
    totals = np.sum(values, axis=1)
    for i, total in enumerate(totals):
        fontsize = dynamic_font_size(total)
        ax.annotate(f'{format_k(total)}',
                    xy=(i, total),
                    xytext=(0, 3),  # Offset text position
                    textcoords="offset points",
                    ha='center', va='bottom', color='black', fontsize=fontsize, fontweight='medium', rotation=90)

    #ax.xaxis.set_major_locator(MultipleLocator(axis_major))
    #ax.xaxis.set_minor_locator(AutoMinorLocator(axis_minor))
    ax.yaxis.set_major_locator(MultipleLocator(axis_major))
    ax.yaxis.set_minor_locator(AutoMinorLocator(axis_minor))

    #from matplotlib.ticker import ScalarFormatter
    #formatter = ScalarFormatter()
    #formatter.set_scientific(True)
    #formatter.set_powerlimits((0, 0))
    #formatter.set_useMathText(True)
    # Apply the formatter to the x-axis
    #ax.xaxis.set_major_formatter(formatter)

    # Add labels and title
    #ax.set_yscale('symlog', linthresh=linthresh)
    ax.set_xlabel('Clients', font_properties=font_properties)
    ax.set_ylabel('Data Size', font_properties=font_properties)
    ax.set_title(title, font_properties=font_properties_title)
    #font_properties['size'] = 16
    legend = ax.legend(prop=font_properties)
    legend.get_frame().set_linewidth(0.0)   

    # Show the plot
    plt.show()

def _barh(ax, 
          num_categories, 
          labels, 
          values, 
          colors, 
          linthresh, 
          u, 
          title: str='',
          font_properties = {
            'family': 'times new roman',
            'size': 12,
            'style': 'normal',
            'weight': 'normal',
            #'variant': 'small-caps'
        },
        font_properties_title = {
            'family': 'times new roman',
            'size': 14,
            'style': 'normal',
            'weight': 'normal',
            #'variant': 'small-caps'
        },
        axis_major: int=10,
        axis_minor: int=5):
    # Plot each category
    left = np.zeros_like(values[:, 0])
    for i in range(num_categories):
        #if f'{u[i]}' == 'CL':
        #print(u[i], values[:, i])
        if i == 0:
            bars = ax.barh(labels, values[:, i], height=0.9, color=colors[i], label=f'{u[i]}')
        else:
            bars = ax.barh(labels, values[:, i], height=0.9, left=np.sum(values[:, :i], axis=1), color=colors[i], label=f'{u[i]}')
        
        left += values[:, i]

        # Add horizontal text annotations for each bar
        for bar in bars:
            width = bar.get_width()
            formatted_width = format_k(width)
            fontsize = dynamic_font_size(width)  
            #if width < 50000:
                # Place text to the right of the bar for small widths
            #    ax.annotate(formatted_width,
            #                xy=(bar.get_x() + width, bar.get_y() + bar.get_height() / 2),
            #                xytext=(2, 0),  # Offset text position
            #                textcoords="offset points",
            #                ha='left', va='center', color='black', fontsize=font_size, rotation=0)
            #else:
            # Place text inside the bar for larger widths
            ax.annotate(formatted_width,
                        xy=(bar.get_x() + width / 2, bar.get_y() + bar.get_height() / 2),
                        xytext=(0, 0),  # Offset text position
                        textcoords="offset points",
                        ha='center', va='center', color='white', fontsize=fontsize, rotation=0)

    # Add total annotations to the right of the bars
    totals = np.sum(values, axis=1)
    for i, total in enumerate(totals):
        formatted_total = format_k(total)
        fontsize = dynamic_font_size(total)
        ax.annotate(f'{formatted_total}',
                    xy=(total, i),
                    xytext=(3, 0),  # Offset text position
                    textcoords="offset points",
                    ha='left', va='center', color='black', fontsize=fontsize, fontweight='medium', rotation=0)

    ax.xaxis.set_major_locator(MultipleLocator(axis_major))
    ax.xaxis.set_minor_locator(AutoMinorLocator(axis_minor))
    ax.tick_params(axis='x', which="major", direction='in', width=1., length=5, labelsize=font_properties['size']-2, top=True, labelfontfamily=font_properties['family'])
    ax.tick_params(axis='x', which="minor", direction='in', width=1., length=3, labelsize=font_properties['size']-2, top=True, labelfontfamily=font_properties['family'])
    ax.tick_params(axis='y', labelsize=font_properties['size']-2, labelfontfamily=font_properties['family'])
    #ax.tick_params(axis='y', which="minor", direction='in', width=1., length=3, labelsize=font_properties['size'], top=True, labelfontfamily=font_properties['family'])

    #ax.yaxis.set_major_locator(MultipleLocator(axis_major))
    #ax.yaxis.set_minor_locator(AutoMinorLocator(axis_minor))

    #from matplotlib.ticker import ScalarFormatter
    #formatter = ScalarFormatter()
    #formatter.set_scientific(True)
    #formatter.set_powerlimits((0, 0))
    #formatter.set_useMathText(True)
    ## Apply the formatter to the x-axis
    
    #ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    # Add labels and title
    #ax.set_xscale('log') # 'symlog', linthresh=linthresh)
    ax.set_xlabel('Data Size', font_properties=font_properties)
    ax.set_ylabel('Clients', font_properties=font_properties)
    ax.set_title(title, font_properties=font_properties_title)
    #font_properties['size'] = 16
    legend = ax.legend(title='Labels', prop=font_properties)
    legend.get_frame().set_linewidth(0.0)   

    #plt.rcParams['legend.title_fontsize'] = font_properties['size'] -2
    # Show the plot
    plt.show()


import math 

def format_func(value, tick_number=None):
    if value == 0:
        return "0"
    
    # Determine the exponent
    exponent = int(math.log10(abs(value)))
    
    # Determine the mantissa
    mantissa = value / (10 ** exponent)
    
    print(value, exponent, mantissa)
    # Format the number in scientific notation
    if exponent % 4 == 0:
        return f"$${{:.2f}}\\times10^{{{exponent}}}$$".format(mantissa)
    else:
        return f"$${{:.2f}}\\times10^{{{exponent - (exponent % 4)}}}$$".format(mantissa)

    #print(value)
    #num_thousands = 0 if abs(value) < 1000 else math.floor (math.log10(abs(value))/3)
    #value = round(value / 1000**num_thousands, 2)
    #return f'{value:g}'+' KMGTPEZY'[num_thousands]