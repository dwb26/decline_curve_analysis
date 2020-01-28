"""
This code is an implementation of using Arp's decline currve as a means of
fitting a curve to some historical well output quantities. Some forecast
data is then used to predict the ultimate recoverable volume from the 
input wellnames.
"""

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from scipy.optimize import least_squares
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


################
# LOAD DATASET #
################
df = pd.read_excel("data_prod.xlsx", sheet_name='PROD_HISTORY')
df_fore = pd.read_excel("data_prod.xlsx", sheet_name='PROD_FORECAST')


##################
# VISUALISE DATA #
##################
def group_wells(wellnames=None):

	"""
	Provides plots of the aggregate output of a collection of input wellnames.
	"""
	
	ax = plt.subplot(111)
	output = ['Gas Prod', 'Oil Prod', 'Water Prod', 'Tubing Pressure', \
	'Casing Pressure', 'Choke Size']

	#### If wellnames == None, extract and plot the grouped dataframe of
	#### aggregate output for all of the wellnames.
	if wellnames == None:
		
		# Sum over all days for each well and each output.
		grouped = df.groupby('Well_allias')[output].sum()

		# Plotting formalities.
		grouped.plot.bar(stacked=True, ax=ax)
		ax.tick_params(axis='x', labelsize=11)
		ax.tick_params(axis='y', labelsize=11)
		ax.legend()
		ax.set_xlabel(xlabel='Well allias', fontdict={'fontsize': 12})
		ax.set_ylabel(ylabel='Output', fontdict={'fontsize': 12})
		ax.set_title('Well allias aggregate output', fontdict={'fontsize': 14})
		return grouped

	#### Extract and plot the grouped dataframe of aggregate output for all of
	#### the input wellnames.

	# Sum over all days for each well and each output.
	hist = df.groupby('Well_allias')[output].sum().loc[wellnames]

	# Plotting formalities.
	hist.plot.bar(stacked=True, ax=ax)
	ax.tick_params(axis='x', labelsize=11)
	ax.tick_params(axis='y', labelsize=11)
	ax.legend()
	ax.set_xlabel(xlabel='Well allias', fontdict={'fontsize': 12})
	ax.set_ylabel(ylabel='Output', fontdict={'fontsize': 12})
	ax.set_title('Well allias aggregate output', fontdict={'fontsize': 14})
	return hist

def monthly_prod(hist):

	"""
	Takes a dataframe of daily time series and plots the monthly and daily 
	aggregate output over all of the wells.
	"""

	# Preliminaries
	ax1 = plt.subplot(211)
	ax2 = plt.subplot(212)
	fig2.subplots_adjust(hspace=.35)
	output = ['Gas Prod', 'Oil Prod', 'Water Prod', 'Tubing Pressure', \
	'Casing Pressure', 'Choke Size']

	# Group data by month and aggregate over outputs.
	monthly = hist.groupby(pd.Grouper(key='Prod Date', freq='M')).sum()
	monthly.index = monthly.index.strftime("%Y-%m")

	# Group data by day and aggregate over outputs.
	daily = hist.groupby(pd.Grouper(key='Prod Date', freq='D')).sum()
	daily = daily.reset_index()

	#### Monthly bar chart plotting.
	monthly.plot(kind='bar', stacked=True, ax=ax1)
	ax1.set_xticklabels([])
	ax1.set_title('Monthly aggregate output', fontdict={'fontsize': 12})
	ax1.legend(fontsize=8)
	ax1.tick_params(axis='y', labelsize=11)

	#### Daily time series plotting.
	[ax2.plot(daily['Prod Date'], daily[quant], linewidth=.25, label=quant) \
	for quant in output]
	leg = ax2.legend(fontsize=8)
	for line in leg.get_lines():
	    line.set_linewidth(4.0)
	ax2.set(xlim=(daily['Prod Date'][0], daily['Prod Date'][4636]))
	ax2.tick_params(axis='x', labelsize=11)
	ax2.tick_params(axis='y', labelsize=11)
	ax2.set_title('Original daily time series aggregate output', \
		fontdict={'fontsize': 12})
	ax2.set_xlabel('Time (years)', fontdict={'fontsize': 12})
	return monthly


#################
# MODEL FITTING #
#################
def model(theta, X):
	"""
	Implementation of Arp's decline curve model. The input parameter theta
	will be learnt from the training data X and initiated with an initial
	guess.
	"""

	[qinit, b_factor, decline_rate] = theta
	yhat = qinit/(1 + b_factor*decline_rate*X)**(1/b_factor)
	return yhat

def cost(theta, X, y):
	"""
	Uses the instances from the training data to tune the theta model parameter
	using the least squares solver.
	"""
	return model(theta, X) - y

def train_model(X, y, theta_init):
	"""
	Minimises the cost function in a least square sense to learn theta that
	provides a best least squares fit of the Arp's decline curve model.
	"""
	theta_opt = least_squares(cost, theta_init, method='lm', args=(X, y), verbose=1)
	return theta_opt

def plot_curves(wellnames=None):

	""" Takes as input wellnames and returns plots of the fitted curve to the 
	time series data of gas, oil and water produced. For example wellnames(['P001']).
	If wellnames == None then None is returned. If len(wellnames) == 1 then the 
	output for that well and the fitted curve is plotted. If len(wellnames > 1) then 
	the wellnames are aggregated by date and their combined output is plotted along 
    with the fitted curve. """

    # Preliminaries.
	output = ['Gas Prod', 'Oil Prod', 'Water Prod']
	colors = ['cornflowerblue', 'forestgreen', 'darkorchid']
	fig3, axs = plt.subplots(2, 3, num=3, figsize=(8, 6))
	plt.tight_layout()
	fig3.subplots_adjust(wspace=.3, hspace=.1)

    # Initial guess for unknown parameter.
	theta_init = np.array([1000, .5, 1])

	if wellnames == None:
		return None

	elif len(wellnames) == 1:

    	# Extract dataframe consisting only of the input wellname.
		df_well = df.loc[df['Well_allias'] == wellnames[0]]
			
		for i in np.arange(len(output)):

			# Prevent error due to NaN values by only inputting non NaN values into 
			# the cost function.
			valid_days = df_well.loc[df_well[output[i]] >= 0]
			t = np.linspace(1, len(df_well.index), len(valid_days))

			# Compute parameter estimation from train_model and compute the absolute 
			# error between the resulting model and the observed data.
			theta_opt = train_model(t, valid_days[output[i]], theta_init)
			err = np.abs(model(theta_opt.x, t) - valid_days[output[i]])

			# Plotting formalities.
			axs[0, i].scatter(df_well['Prod Date'], df_well[output[i]], s=2, \
				color=colors[i])
			axs[0, i].plot(valid_days['Prod Date'], model(theta_opt.x, t), 'orange',
				linewidth=.5)
			axs[1, i].plot(valid_days['Prod Date'], err, color=colors[i], linewidth=.5)
			axs[0, i].tick_params(axis='y', labelsize=9)
			axs[1, i].tick_params(axis='x', labelsize=9)
			axs[1, i].tick_params(axis='y', labelsize=9)
			axs[0, i].xaxis.set_major_formatter(plt.NullFormatter())
			axs[0, i].set_title(wellnames[0] + ' ' + output[i] + \
				' (top) and error (below) vs. time', fontdict={'fontsize': 11})

	else:
		# Aggregate the output over each day for each of the input wells.
		daily = df.loc[df['Well_allias'].isin(wellnames)]
		daily = daily.groupby(pd.Grouper(key='Prod Date', freq='D')).sum()
		daily = daily.reset_index()

		for i in np.arange(len(output)):

			# Prevent error due to NaN values by only inputting non NaN values into 
			# the cost function.
			valid_days = daily.loc[daily[output[i]] >= 0 ]
			t = np.linspace(1, len(daily.index), len(valid_days))

			# Compute parameter estimation from train_model and compute the absolute 
			# error between the resulting model and the observed data.
			theta_opt = train_model(t, valid_days[output[i]], theta_init)
			err = np.abs(model(theta_opt.x, t) - valid_days[output[i]])

			# Plotting formalities.
			axs[0, i].scatter(daily['Prod Date'], daily[output[i]], s=2, \
				color=colors[i])
			axs[0, i].plot(valid_days['Prod Date'], model(theta_opt.x, t), 'orange',\
				linewidth=.5)
			axs[1, i].plot(valid_days['Prod Date'], err, color=colors[i], linewidth=.5)
			axs[0, i].tick_params(axis='y', labelsize=9)
			axs[1, i].tick_params(axis='x', labelsize=9)
			axs[1, i].tick_params(axis='y', labelsize=9)
			axs[0, i].xaxis.set_major_formatter(plt.NullFormatter())
			axs[0, i].set_title('Grouped ' + output[i] + \
				' (top) and error (below) vs. time', fontdict={'fontsize': 11})
	return fig3

def cumul_time(wellnames): 
    """ 
	N_p = cumulative production based on the recorded data.
	EUR = ultimate recoverable volume.

    The purpose of this function is to quantify N_p and EUR on a bar plot. The x-axis 
    denotes the input wells and the N_p and EUR outputs levels are plotted for each
    of the input wells. Three subplots can then be generated for each of gas, oil and 
    water to measure the EUR compared to N_p for each of the wells.
    
    The data is such that the last day of production and the date of abandonment is 
    the same for each well, however the code is written to handle data for which these 
    dates are different.
    """
    
    if wellnames == None:
        return None
    
    # Extract the historical and forecast dataframes for the input wellnames.
    well_hist_df = df.loc[df['Well_allias'].isin(wellnames)].set_index('Well_allias')
    well_fore_df = df_fore.loc[df_fore['Well_allias'].isin(wellnames)]\
    .set_index('Well_allias')
    
    # Aggregate historical output for gas, water and oil for each wellname.
    output = ['Gas Prod', 'Oil Prod', 'Water Prod']
    Np_df = well_hist_df.groupby('Well_allias').sum()[output]
    
    #### Construct dataframe of output values for each of the wells on the last day of 
    # historical data.
    d1 = {}; tab = []
    
    for well in wellnames:
        
        # Get the last date of production for the well iterate.
        tf = well_hist_df.loc[well].iloc[-1, well_hist_df.columns.get_loc('Prod Date')]
        
        # Access the output values of the well iterate on this date.
        values = well_hist_df.loc[well].set_index('Prod Date').loc[tf]
        
        # Append these values in a dict where key is the well_allias and value is a 
        # list of the output values.
        d1[well] = [values[output][i] for i in np.arange(len(output))]
            
        # Extract the time of abandonment from the forecast data for each well.
        tab.append(len(well_hist_df.index) + len(well_fore_df.index))
    
    # Data from of the output values on the last day of production for each input well. 
    # Indices are the well_alliasand the columns are the outputs (by virtue of the 
    # transpose operation).
    qf_df = pd.DataFrame(data=d1, index=output).T
    
    #### Run train_model() to compute parameter estimations for each well to compute q_ab 
    #### and Q_f for each output.
    
    # Initial guess for unknown parameter.
    theta_init = np.array([1000, .5, 1])
    d2sub = {}; d2 = {}; m = 0
    
    #### Generate a dict where each key is a well allias with value a subdict, where each 
    #### key of the subdict is an output type (e.g. Gas Prod) and each key is the value 
    #### of Qf corresponding to the Arps parameter values.
    for well in wellnames:
        
        for i in np.arange(len(output)):
            
            # Only operate on columns that contain non NaN values.
            valid_days = well_hist_df.loc[well_hist_df[output[i]] >= 0]
            t = np.linspace(1, len(valid_days), len(valid_days))
            
            # Invoke train_model() to compute the Arps parameters for the output of this 
            # well.
            [q_init, b_factor, decline_rate] = train_model(t, valid_days[output[i]], \
            	theta_init).x
            
            # Compute Q_f for each output of the well iterate.
            qab = q_init/(1 + b_factor*decline_rate*tab[m])**(1/b_factor)
            qf = qf_df[output[i]].loc[well]
            d2sub[output[i]] = q_init**b_factor/((1 - b_factor)*decline_rate)*\
            (qf**(1 - b_factor) - qab**(1 - b_factor))
                                                                               
        # Store this dict under a key for the well in a super dict.
        d2[well] = d2sub
        d2sub = {}
        m += 1
    
    #### Use this dict to generate a dataframe for Qf, where each index is well_allias 
    #### and each column is Qf for that output.
    Qf_data = [d2[wells] for wells in wellnames]
    Qf_df = pd.DataFrame(data=[Qf_data[i] for i in np.arange(len(Qf_data))], \
    	index=wellnames)
    
    #### Now combine Np and Qf dataframes to create EUR and subsequent output dataframes.
    EUR_df = Np_df.append(Qf_df).reset_index().groupby('index').sum().\
    rename_axis('Well_allias')
    gas_df = pd.DataFrame({'Np': Np_df['Gas Prod'], 'EUR': EUR_df['Gas Prod']}, \
    	index=wellnames)
    oil_df = pd.DataFrame({'Np': Np_df['Oil Prod'], 'EUR': EUR_df['Oil Prod']}, \
    	index=wellnames)
    water_df = pd.DataFrame({'Np': Np_df['Water Prod'], 'EUR': EUR_df['Water Prod']}, \
    	index=wellnames)
    
    #### Plotting.
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)
    gas_df.plot.bar(rot=0, ax=ax1)
    oil_df.plot.bar(rot=0, ax=ax2)
    water_df.plot.bar(rot=0, ax=ax3)
    ax1.set_title('Gas Prod', fontdict={'fontsize': 14})
    ax2.set_title('Oil Prod', fontdict={'fontsize': 14})
    ax3.set_title('Water Prod', fontdict={'fontsize': 14})
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    ax3.legend(fontsize=8)
    ax1.set_xlabel('Well allias', fontdict={'fontsize': 12})
    ax2.set_xlabel('Well allias', fontdict={'fontsize': 12})
    ax3.set_xlabel('Well allias', fontdict={'fontsize': 12})
    ax1.tick_params(axis='y', labelsize=11)
    ax2.tick_params(axis='y', labelsize=11)
    ax3.tick_params(axis='y', labelsize=11)
    return Np_df, EUR_df


wellnames = ['P001', 'P002', 'P010', 'P013']
fig1 = plt.figure(1, figsize=(8, 6))
group_wells(wellnames)
fig1.show()
fig2 = plt.figure(2, figsize=(8, 6))
monthly_prod(df)
fig2.show()
fig3 = plot_curves(['P003']) # Results for this routine are clearer with one input well
							 # but can also be run with multiple inputs like the other
							 # functions.
fig3.show()
fig4 = plt.figure(4, figsize=(8, 6))
cumul_time(wellnames)
fig4.show()
input()