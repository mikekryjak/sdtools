import os
from textwrap import fill
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.sankey import Sankey
from matplotlib import ticker, cm
import bisect
from matplotlib import animation
from matplotlib import colors
from scipy.interpolate import interp1d

el_mass = 9.10938e-31
ion_mass = 2 * 1.67262e-27
epsilon_0 = 8.854188E-12
el_charge = 1.602189E-19
boltzmann_k = 1.38064852E-23
bohr_radius = 5.291772e-11
planck_h = 6.62607004e-34


class SKComp:
    def __init__(self, rundecks, labels):
        self.rundecks = rundecks
        self.labels = labels
        self.linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
        # self.linestyles = ['solid', 'dotted', 'dashed',
        #                    'dashdot', 'loosely dotted', 'loosely dashdotdotted']
        self.get_line_colours()
        self.build_legend()

    def get_line_colours(self):

        cmap = plt.cm.get_cmap('plasma')
        if self.rundecks[0].sort_by == 'density':

            min_dens = self.rundecks[0].avg_densities[0]
            max_dens = self.rundecks[0].avg_densities[-1]
            for rundeck in self.rundecks:
                if min(rundeck.avg_densities) < min_dens:
                    min_dens = min(rundeck.avg_densities)
                if max(rundeck.avg_densities) > max_dens:
                    max_dens = max(rundeck.avg_densities)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
                vmin=min_dens, vmax=max_dens))

        elif self.rundecks[0].sort_by == 'power':

            min_p = self.rundecks[0].input_el_powers[0]
            max_p = self.rundecks[0].input_el_powers[-1]
            for rundeck in self.rundecks:
                if min(rundeck.avg_densities) < min_p:
                    min_p = min(rundeck.input_el_powers)
                if max(rundeck.avg_densities) > max_p:
                    max_p = max(rundeck.input_el_powers)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
                vmin=self.min_p, vmax=self.max_p))

        self.line_colours = [None] * len(self.rundecks)
        for i, rundeck in enumerate(self.rundecks):
            self.line_colours[i] = [None] * rundeck.num_sims
            for j in range(rundeck.num_sims):
                if rundeck.sort_by == 'density':
                    self.line_colours[i][j] = sm.to_rgba(
                        rundeck.avg_densities[j])
                elif rundeck.sort_by == 'power':
                    self.line_colours[i][j] = sm.to_rgba(
                        rundeck.input_el_powers[j])

    def build_legend(self):

        self.legend_lines = []
        self.legend_labels = []
        for j, run in enumerate(self.rundecks[0].runs):
            if self.rundecks[0].sort_by == 'density':
                profile_label = r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                    self.rundecks[0].avg_densities[j]/1e19) + r'$\times 10^{19}$m$^{-3}$'
                self.legend_lines.append(
                    Line2D([0], [0], color=self.line_colours[0][j]))
                self.legend_labels.append(profile_label)
            elif self.rundecks[0].sort_by == 'power':
                profile_label = r'$P_{in}$' + ' = {:1.1f}'.format(
                    self.rundecks[0].input_el_powers[j]) + r'MWm$^{-2}$'
                self.legend_lines.append(
                    Line2D([0], [0], color=self.line_colours[0][j]))
                self.legend_labels.append(profile_label)

        self.legend_lines = [self.legend_lines[0], self.legend_lines[int(
            len(self.legend_lines)/2)], self.legend_lines[-1]]
        self.legend_labels = [self.legend_labels[0], self.legend_labels[int(
            len(self.legend_labels)/2)], self.legend_labels[-1]]

        for i in range(len(self.rundecks)):
            self.legend_lines.append(
                Line2D([0], [0], linestyle=self.linestyles[i], color='black'))
            self.legend_labels.append(self.labels[i])

    def flux_rollover_plot(self, sort_by='density'):
        if sort_by == 'density':

            self.upstream_densities = [None] * len(self.rundecks)
            self.avg_densities = [None] * len(self.rundecks)
            self.target_fluxes = [None] * len(self.rundecks)
            self.target_temps = [None] * len(self.rundecks)

            for i, rundeck in enumerate(self.rundecks):
                rundeck.sort_by_density()
                self.upstream_densities[i], self.avg_densities[i] = rundeck.get_densities(
                )
                self.target_fluxes[i], self.target_temps[i] = rundeck.get_target_conditions(
                )

            fig, ax = plt.subplots(1)
            ax2 = ax.twinx()
            for i, rundeck in enumerate(self.rundecks):
                ax.plot(self.avg_densities[i],
                        self.target_temps[i], self.linestyles[i], color='red')
                ax2.plot(self.avg_densities[i], self.target_fluxes[i],
                         self.linestyles[i], color='black', label=self.labels[i])

            ax2.legend()
            ax2.set_ylabel('Target ion flux [m$^{-2}$s$^{-1}$]')
            ax.set_ylabel('Target electron temperature [eV]', color='red')
            ax.tick_params(axis='y', colors='red')
            ax.set_xlabel(r'Line-averaged density [m$^{-3}$]')

            fig.tight_layout()

    def target_heat_flux_plot(self, sort_by='density', species='electrons'):

        fig, ax = plt.subplots(figsize=(5, 4.0))
        fig.tight_layout(pad=2.0)
        linestyles = ['-', '-', '-', '-', '--', '--', '--', '--']
        colours = ['red', 'blue', 'green', 'black',
                   'red', 'blue', 'green', 'black']
        labels = ['4MWm$^{-2}$', '8MWm$^{-2}$',
                  '16MWm$^{-2}$', '64MWm$^{-2}$', '', '', '', '']
        if sort_by == 'density':
            self.upstream_densities = [None] * len(self.rundecks)
            self.avg_densities = [None] * len(self.rundecks)
            q_sh = [None] * len(self.rundecks)

            for i, rundeck in enumerate(self.rundecks):
                rundeck.sort_by_density()
                self.upstream_densities[i], self.avg_densities[i] = rundeck.get_densities(
                )

                q_sh[i] = [None] * len(rundeck.runs)
                for j, r in enumerate(rundeck.runs):
                    if species == 'electrons':
                        q_sh[i][j] = 1e-6 * r.q_sh_e()

                ax.plot(self.avg_densities[i], q_sh[i], color=colours[i],
                        linestyle=linestyles[i], label=labels[i])

            ax.set_xlabel('Line-averaged density [m$^{-3}$]')
            ax.set_ylabel('$q_{sh,e}$ [MWm$^{-2}$]')

            ax.legend()

    def gamma_plot(self, species='electrons'):

        fig, ax = plt.subplots(figsize=(5, 4.0))
        fig.tight_layout(pad=2.0)
        linestyles = ['-', '-', '-', '-', '--', '--', '--', '--']
        colours = ['red', 'blue', 'green', 'black',
                   'red', 'blue', 'green', 'black']
        labels = ['4MWm$^{-2}$', '8MWm$^{-2}$',
                  '16MWm$^{-2}$', '64MWm$^{-2}$', '', '', '', '']
        if self.rundecks[0].sort_by == 'density':
            self.upstream_densities = [None] * len(self.rundecks)
            self.avg_densities = [None] * len(self.rundecks)
            gamma = [None] * len(self.rundecks)
            gamma_pred = [None] * len(self.rundecks)

            for i, rundeck in enumerate(self.rundecks):
                rundeck.sort_by_density()
                self.upstream_densities[i], self.avg_densities[i] = rundeck.get_densities(
                )

                gamma[i] = [None] * len(rundeck.runs)
                gamma_pred[i] = [None] * len(rundeck.runs)
                for j, r in enumerate(rundeck.runs):
                    if species == 'electrons':
                        if r.full_fluid and (not r.intrinsic_coupling):
                            gamma[i][j] = 2.0 - 0.5 * np.log(2.0*np.pi*(1.0 + (
                                r.data['ION_TEMPERATURE'][-1]/r.data['TEMPERATURE'][-1])) * (el_mass/(1.0*ion_mass)))

                            a = 12.638
                            b = -1.239
                            c = 2.053
                            d = -0.368
                            Tu = r.data['TEMPERATURE'][0]
                            Tt = r.data['TEMPERATURE'][-1]
                            nu_star = r.get_collisionality()
                            L = r.connection_length
                            ion_flux, _ = r.get_target_conditions()
                            ion_flux = ion_flux / 1e23
                            gamma_pred[i][j] = gamma[i][j] * (1.0 + a * (nu_star ** b) * (
                                (r.T_norm*(Tu-Tt)/L) ** c) * (ion_flux ** d))

                        else:
                            r.load_vars('SHEATH_DATA')
                            gamma[i][j] = r.data['SHEATH_DATA'][0]

                ax.plot(self.avg_densities[i], gamma[i], color=colours[i],
                        linestyle=linestyles[i], label=labels[i])
                # ax.plot(self.upstream_densities[i], gamma[i], color=colours[i],
                #         linestyle=linestyles[i], label=labels[i])

        # for i in range(4):
        #     ax.plot(self.avg_densities[i], gamma_pred[i], color=colours[i],
        #                 linestyle='--',marker='x')

        ax.set_ylabel('$\gamma_e$')
        ax.set_xlabel('Line-averaged density [m$^{-3}$]')
        print(gamma_pred)
        ax.legend()

    def ionization_cost_plot(self, sort_by='density'):

        fig, ax = plt.subplots()
        ax.set_xlabel('$T_{e,t}$ [eV]')
        ax.set_ylabel('$E_{iz}$ [eV]')

        self.target_fluxes = [None] * len(self.rundecks)
        self.target_temps = [None] * len(self.rundecks)
        E_iz = [None] * len(self.rundecks)
        for i, rundeck in enumerate(self.rundecks):
            rundeck.sort_by_density()
            self.target_fluxes[i], self.target_temps[i] = rundeck.get_target_conditions(
            )
            E_iz[i] = np.zeros(len(rundeck.runs))

            for j, r in enumerate(rundeck.runs):
                input_powers, q_E, q_sh_e, q_sh_i, q_sh_n, q_recyc_n, q_n_e, q_ion_i, q_rec_i, q_rad, q_ie, q_cx = r.get_energy_channels()
                E_iz[i][j] = (1e6 / el_charge) * (q_n_e + q_rad) / \
                    self.target_fluxes[i][j]

            ax.plot(self.target_temps[i], E_iz[i],
                    self.linestyles[i], label=self.labels[i])

        ax.legend()

    def target_plot(self, variable='DENSITY', sort_by='density'):
        if sort_by == 'density':
            self.rundeck1.sort_by_density()
            self.rundeck2.sort_by_density()
        elif sort_by == 'power':
            self.rundeck1.sort_by_power()
            self.rundeck2.sort_by_power()

        if variable == 'DENSITY':
            n_e_boundary1 = np.zeros(len(self.rundeck1.runs))
            densities1 = np.zeros(len(self.rundeck1.runs))
            n_e_boundary2 = np.zeros(len(self.rundeck2.runs))
            densities2 = np.zeros(len(self.rundeck2.runs))
            for i, r in enumerate(self.rundeck1.runs):
                densities1[i] = r.upstream_density
                n_e_boundary1[i] = r.get_target_density() * r.n_norm
            for i, r in enumerate(self.rundeck2.runs):
                densities2[i] = r.upstream_density
                n_e_boundary2[i] = r.get_target_density() * r.n_norm

            fig, ax = plt.subplots(1)
            ax.plot(densities1, n_e_boundary1, '-', label=self.label1)
            ax.plot(densities2, n_e_boundary2, '--', label=self.label2)
            ax.set_xlabel('Upstream density [m$^{-3}$]')
            ax.set_ylabel('Target density [m$^{-3}$]')
            ax.legend()

        elif variable == 'TEMPERATURE':
            T_e_boundary1 = np.zeros(len(self.rundeck1.runs))
            densities1 = np.zeros(len(self.rundeck1.runs))
            T_e_boundary2 = np.zeros(len(self.rundeck2.runs))
            densities2 = np.zeros(len(self.rundeck2.runs))
            for i, r in enumerate(self.rundeck1.runs):
                densities1[i] = r.upstream_density
                T_e_boundary1[i] = r.data['TEMPERATURE'][-1] * r.T_norm
            for i, r in enumerate(self.rundeck2.runs):
                densities2[i] = r.upstream_density
                T_e_boundary2[i] = r.data['TEMPERATURE'][-1] * r.T_norm

            fig, ax = plt.subplots(1)
            ax.plot(densities1, T_e_boundary1, '-', label=self.label1)
            ax.plot(densities2, T_e_boundary2, '--', label=self.label2)
            ax.set_xlabel('Upstream density [m$^{-3}$]')
            ax.set_ylabel('Target electron temperature [eV]')
            ax.legend()

    def energy_discrepancy_plot(self, sort_by='density'):
        if sort_by == 'density':
            self.rundeck1.sort_by_density()
            self.rundeck2.sort_by_density()
        elif sort_by == 'power':
            self.rundeck1.sort_by_power()
            self.rundeck2.sort_by_power()

        if self.rundeck1.runs[0].neut_mom_eqn is False and self.rundeck1.runs[0].ion_temp_eqn is False:
            discrepancies1 = np.zeros(len(self.rundeck1.runs))
            discrepancies2 = np.zeros(len(self.rundeck2.runs))
            input_powers1 = np.zeros(len(self.rundeck1.runs))
            input_powers2 = np.zeros(len(self.rundeck2.runs))
            for i, r in enumerate(self.rundeck1.runs):
                input_powers1[i], q_E, q_sh_e, q_sh_i, q_sh_n, q_recyc_n, q_n_e, q_ion_i, q_rec_i, q_rad, q_ie, q_cx = r.get_energy_channels()
                discrepancies1[i] = (input_powers1[i] -
                                     (q_sh_e + q_n_e + q_rad + q_E))
            for i, r in enumerate(self.rundeck2.runs):
                input_powers2[i], q_E, q_sh_e, q_sh_i, q_sh_n, q_recyc_n, q_n_e, q_ion_i, q_rec_i, q_rad, q_ie, q_cx = r.get_energy_channels()
                discrepancies2[i] = (input_powers1[i] -
                                     (q_sh_e + q_n_e + q_rad + q_E))

            fix, ax = plt.subplots(1)
            if sort_by == 'power':
                ax.plot(input_powers1, discrepancies1, '-', label=self.label1)
                ax.plot(input_powers2, discrepancies2, '--', label=self.label2)
                ax.set_ylabel('Discrepancy [MW$m^{-2}$]')
                ax.set_xlabel('Input power [MW$m^{-2}$]')
                ax.set_title(r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                    self.rundeck1.runs[0].avg_density/1e19) + r'$\times 10^{19}$m$^{-3}$', loc='left')
                ax.legend()
            elif sort_by == 'density':
                _, densities1 = self.rundeck1.get_densities()
                _, densities2 = self.rundeck2.get_densities()
                ax.plot(densities1, discrepancies1, '-', label=self.label1)
                ax.plot(densities2, discrepancies2, '--', label=self.label2)
                ax.set_ylabel('Discrepancy [MW$m^{-2}$]')
                ax.set_xlabel(r'$\langle n_{heavy} \rangle$' + ' [m$^{-3}$]')
                ax.set_title(
                    '$P_{in} = $' + str(input_powers1[0]) + 'MWm$^{-2}$', loc='left')
                ax.legend()

        if self.rundeck1.runs[0].neut_mom_eqn is False and self.rundeck1.runs[0].ion_temp_eqn is True:
            el_discrepancies1 = np.zeros(len(self.rundeck1.runs))
            el_discrepancies2 = np.zeros(len(self.rundeck2.runs))
            ion_discrepancies1 = np.zeros(len(self.rundeck1.runs))
            ion_discrepancies2 = np.zeros(len(self.rundeck2.runs))
            input_powers1 = np.zeros(len(self.rundeck1.runs))
            input_powers2 = np.zeros(len(self.rundeck2.runs))

            for i, run in enumerate(self.rundeck1.runs):
                input_powers1[i], q_E, q_sh_e, q_sh_i, q_sh_n, q_recyc_n, q_n_e, q_ion_i, q_rec_i, q_rad, q_ie, q_cx = run.get_energy_channels()
                el_discrepancies1[i] = input_powers1[i] / \
                    2 + q_ie - (q_sh_e + q_n_e + q_rad + q_E)
                ion_discrepancies1[i] = input_powers1[i]/2 + \
                    q_E + q_ion_i - (q_ie + q_rec_i + q_sh_i)

            for i, run in enumerate(self.rundeck2.runs):
                input_powers2[i], q_E, q_sh_e, q_sh_i, q_sh_n, q_recyc_n, q_n_e, q_ion_i, q_rec_i, q_rad, q_ie, q_cx = run.get_energy_channels()
                el_discrepancies2[i] = input_powers2[i] / \
                    2 + q_ie - (q_sh_e + q_n_e + q_rad + q_E)
                ion_discrepancies2[i] = input_powers2[i]/2 + \
                    q_E + q_ion_i - (q_ie + q_rec_i + q_sh_i)

            fix, ax = plt.subplots(1)

            if sort_by == 'power':
                ax.plot(input_powers1, el_discrepancies1,
                        '-', color='red', label='electrons')
                ax.plot(input_powers2, el_discrepancies2, '--', color='red')
                ax.plot(input_powers1, ion_discrepancies1,
                        '-', color='blue', label='ions')
                ax.plot(input_powers2, ion_discrepancies2, '--', color='blue')
                ax.plot(input_powers1, el_discrepancies1 +
                        ion_discrepancies1, '-', color='black', label='total')
                ax.plot(input_powers2, el_discrepancies2 +
                        ion_discrepancies2, '--', color='black')
                ax.set_ylabel('Discrepancy [MW$m^{-2}$]')
                ax.set_xlabel('Input power [MW$m^{-2}$]')
                ax.set_title(r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                    self.rundeck1.runs[0].avg_density/1e19) + r'$\times 10^{19}$m$^{-3}$', loc='left')
                ax.legend()

            elif sort_by == 'density':
                _, densities1 = self.rundeck1.get_densities()
                _, densities2 = self.rundeck2.get_densities()
                ax.plot(densities1, el_discrepancies1, '-',
                        color='red', label='electrons')
                ax.plot(densities2, el_discrepancies2, '--', color='red')
                ax.plot(densities1, ion_discrepancies1,
                        '-', color='blue', label='ions')
                ax.plot(densities2, ion_discrepancies2, '--', color='blue')
                ax.plot(densities1, el_discrepancies1 +
                        ion_discrepancies1, '-', color='black', label='total')
                ax.plot(densities2, el_discrepancies2 +
                        ion_discrepancies2, '--', color='black')
                ax.set_ylabel('Discrepancy [MW$m^{-2}$]')
                ax.set_xlabel(r'$\langle n_{heavy} \rangle$' + ' [m$^{-3}$]')
                ax.set_title(
                    '$P_{in} = $' + str(input_powers1[0]) + 'MWm$^{-2}$', loc='left')
                ax.legend()

        if self.rundeck1.runs[0].neut_mom_eqn is True and self.rundeck1.runs[0].ion_temp_eqn is True:
            el_discrepancies1 = np.zeros(len(self.rundeck1.runs))
            el_discrepancies2 = np.zeros(len(self.rundeck2.runs))
            ion_discrepancies1 = np.zeros(len(self.rundeck1.runs))
            ion_discrepancies2 = np.zeros(len(self.rundeck2.runs))
            neut_discrepancies1 = np.zeros(len(self.rundeck1.runs))
            neut_discrepancies2 = np.zeros(len(self.rundeck2.runs))
            input_powers1 = np.zeros(len(self.rundeck1.runs))
            input_powers2 = np.zeros(len(self.rundeck2.runs))
            for i, run in enumerate(self.rundeck1.runs):
                input_powers1[i], q_E, q_sh_e, q_sh_i, q_sh_n, q_recyc_n, q_n_e, q_ion_i, q_rec_i, q_rad, q_ie, q_cx = run.get_energy_channels()
                el_discrepancies1[i] = input_powers1[i] / \
                    2 + q_ie - (q_sh_e + q_n_e + q_rad + q_E)
                ion_discrepancies1[i] = input_powers1[i]/2 + \
                    q_E + q_ion_i - (q_ie + q_rec_i + q_sh_i + q_cx)
                neut_discrepancies1[i] = q_cx + q_rec_i + \
                    q_recyc_n - (q_ion_i + q_sh_n)
            for i, run in enumerate(self.rundeck2.runs):
                input_powers2[i], q_E, q_sh_e, q_sh_i, q_sh_n, q_recyc_n, q_n_e, q_ion_i, q_rec_i, q_rad, q_ie, q_cx = run.get_energy_channels()
                el_discrepancies2[i] = input_powers2[i] / \
                    2 + q_ie - (q_sh_e + q_n_e + q_rad + q_E)
                ion_discrepancies2[i] = input_powers2[i]/2 + \
                    q_E + q_ion_i - (q_ie + q_rec_i + q_sh_i + q_cx)
                neut_discrepancies2[i] = q_cx + q_rec_i + \
                    q_recyc_n - (q_ion_i + q_sh_n)

            fix, ax = plt.subplots(1)
            if sort_by == 'power':
                ax.plot(input_powers1, el_discrepancies1,
                        '-', color='red', label='electrons')
                ax.plot(input_powers2, el_discrepancies2, '--', color='red')
                ax.plot(input_powers1, ion_discrepancies1,
                        '-', color='blue', label='ions')
                ax.plot(input_powers2, ion_discrepancies2, '--', color='blue')
                ax.plot(input_powers1, neut_discrepancies1,
                        '-', color='green', label='neutrals')
                ax.plot(input_powers2, neut_discrepancies2,
                        '--', color='green')
                ax.plot(input_powers1, el_discrepancies1+ion_discrepancies2,
                        '-', color='black', label='total plasma')
                ax.plot(input_powers2, el_discrepancies2 +
                        ion_discrepancies2, '--', color='black')
                ax.set_ylabel('Discrepancy [MW$m^{-2}$]')
                ax.set_xlabel('Input power [MW$m^{-2}$]')
                ax.set_title(r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                    self.rundeck1.runs[0].avg_density/1e19) + r'$\times 10^{19}$m$^{-3}$', loc='left')
                ax.legend()
            elif sort_by == 'density':
                _, densities1 = self.rundeck1.get_densities()
                _, densities2 = self.rundeck2.get_densities()
                ax.plot(densities1, el_discrepancies1, '-',
                        color='red', label='electrons')
                ax.plot(densities2, el_discrepancies2, '--', color='red')
                ax.plot(densities1, ion_discrepancies1,
                        '-', color='blue', label='ions')
                ax.plot(densities2, ion_discrepancies2, '--', color='blue')
                ax.plot(densities1, neut_discrepancies1,
                        '-', color='green', label='neutrals')
                ax.plot(densities2, neut_discrepancies2, '--', color='green')
                ax.plot(densities1, el_discrepancies1+ion_discrepancies1,
                        '-', color='black', label='total plasma')
                ax.plot(densities2, el_discrepancies2 +
                        ion_discrepancies2, '--', color='black')
                ax.set_ylabel('Discrepancy [MW$m^{-2}$]')
                ax.set_xlabel(r'$\langle n_{heavy} \rangle$' + ' [m$^{-3}$]')
                ax.set_title(
                    '$P_{in} = $' + str(input_powers1[0]) + 'MWm$^{-2}$', loc='left')
                ax.legend()

    def profile(self, variable, sort_by='density', cells=(0, None)):
        if sort_by == 'power':
            for rundeck in self.rundecks:
                rundeck.sort_by_power()
        elif sort_by == 'density':
            for rundeck in self.rundecks:
                rundeck.sort_by_density()

        fig, ax = plt.subplots(1)

        to_plot = [0,2]
        for i, rundeck in enumerate(self.rundecks):
            for j, r in enumerate(rundeck.runs):
                if j in to_plot:
                    norm_value = r.get_var_norm(variable)
                    y_label = r.get_var_label(variable)
                    r.load_vars(variable)

                    if variable == 'NEUTRAL_DENS':
                        if r.num_neutrals > 1:
                            plot_data = np.sum(r.data[variable], 1) * norm_value
                        else:
                            plot_data = r.data[variable] * norm_value
                    else:
                        plot_data = r.data[variable] * norm_value

                    ax.plot(r.xgrid[cells[0]:cells[1]], plot_data[cells[0]:cells[1]],
                            self.linestyles[i], color=self.line_colours[i][j])

        ax.set_title(variable)
        ax.legend(self.legend_lines, self.legend_labels)
        ax.set_ylabel(y_label)
        ax.set_xlabel('x [m]')
        # ax.set_yscale('log')
        fig.tight_layout()

    def radiated_power_plot(self, sort_by='density'):

        # Radiated power comparison
        if sort_by == 'density':
            densities = [None] * len(self.rundecks)
            P_rad = [None] * len(self.rundecks)
            for i, rundeck in enumerate(self.rundecks):
                rundeck.sort_by_density()
                densities[i], _ = rundeck.get_densities()
                P_rad[i] = np.zeros(len(rundeck.runs))
                for j, r in enumerate(rundeck.runs):
                    input_power, q_E, q_sh_e, q_sh_i, q_sh_n, q_recyc_n, q_n_e, q_ion_i, q_rec_i, q_rad, q_ie, q_cx = r.get_energy_channels()
                    P_rad[i][j] = q_rad
        elif sort_by == 'power':

            P_in = [None] * len(self.rundecks)
            P_rad = [None] * len(self.rundecks)
            for i, rundeck in enumerate(self.rundecks):
                rundeck.sort_by_power()
                P_in[i] = np.zeros(len(rundeck.runs))
                P_rad[i] = np.zeros(len(rundeck.runs))
                for j, r in enumerate(rundeck.runs):
                    input_power, q_E, q_sh_e, q_sh_i, q_sh_n, q_recyc_n, q_n_e, q_ion_i, q_rec_i, q_rad, q_ie, q_cx = r.get_energy_channels()
                    P_rad[i][j] = q_rad
                    P_in[i][j] = input_power

        fig, ax = plt.subplots()

        ax.set_ylabel('Radiated power [MWm$^{-2}$]')
        if sort_by == 'density':
            ax.set_xlabel('Upstream density [m$^{-3}$]')
            for i in range(len(self.rundecks)):
                ax.plot(densities[i], P_rad[i], self.linestyles[i],
                        color='black', label=self.labels[i])
        elif sort_by == 'power':
            ax.set_xlabel('Input power [MWm$^{-2}$]')
            for i in range(len(self.rundecks)):
                ax.plot(P_in[i], P_rad[i], self.linestyles[i],
                        color='black', label=self.labels[i])
        ax.legend()

    def ion_rate(self, compare_sd1d=True, sort_by='density'):

        if sort_by == 'power':
            self.rundeck1.sort_by_power()
            self.rundeck2.sort_by_power()
        elif sort_by == 'density':
            self.rundeck1.sort_by_density()
            self.rundeck2.sort_by_density()

        fig, ax = plt.subplots(1)
        cmap = plt.cm.get_cmap('plasma')
        if sort_by == 'power':
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
                vmin=self.rundeck1.runs[0].input_power, vmax=self.rundeck1.runs[-1].input_power))
        elif sort_by == 'density':
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
                vmin=self.rundeck1.runs[0].avg_density, vmax=self.rundeck1.runs[-1].avg_density))

        legend_lines = []
        legend_labels = []
        for r in self.rundeck1.runs:
            r.load_vars(['S_ION_SK'])

            S_ion = r.n_norm * (r.v_th / r.x_norm) * r.data['S_ION_SK']
            try:
                n_n = np.sum(r.data['NEUTRAL_DENS'], 1) * r.n_norm
            except(np.AxisError):
                n_n = r.data['NEUTRAL_DENS'] * r.n_norm

            n_e = r.data['DENSITY'] * r.n_norm

            T_e = r.data['TEMPERATURE'] * r.T_norm

            sk_ion_rate = (S_ion / (n_n * n_e))

            if sort_by == 'density':
                profile_label = r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                    r.avg_density/1e19) + r'$\times 10^{19}$m$^{-3}$'
                c = sm.to_rgba(r.avg_density)
            elif sort_by == 'power':
                profile_label = r'$P_{in}$' + ' = {:1.1f}'.format(
                    r.input_power) + r'MWm$^{-2}$'
                c = sm.to_rgba(r.input_power)

            if sort_by == 'power':
                legend_lines.append(
                    Line2D([0], [0], color=sm.to_rgba(r.input_power)))
            elif sort_by == 'density':
                legend_lines.append(
                    Line2D([0], [0], color=sm.to_rgba(r.avg_density)))
            legend_labels.append(profile_label)

            sk_ion_rate[sk_ion_rate == 0] = np.nan

            ax.plot(T_e, sk_ion_rate, label=profile_label, color=c)

        for r in self.rundeck2.runs:
            r.load_vars(['S_ION_SK'])

            S_ion = r.n_norm * (r.v_th / r.x_norm) * r.data['S_ION_SK']
            try:
                n_n = np.sum(r.data['NEUTRAL_DENS'], 1) * r.n_norm
            except(np.AxisError):
                n_n = r.data['NEUTRAL_DENS'] * r.n_norm

            n_e = r.data['DENSITY'] * r.n_norm

            T_e = r.data['TEMPERATURE'] * r.T_norm

            sk_ion_rate = (S_ion / (n_n * n_e))

            if sort_by == 'density':
                profile_label = r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                    r.avg_density/1e19) + r'$\times 10^{19}$m$^{-3}$'
                c = sm.to_rgba(r.avg_density)
            elif sort_by == 'power':
                profile_label = r'$P_{in}$' + ' = {:1.1f}'.format(
                    r.input_power) + r'MWm$^{-2}$'
                c = sm.to_rgba(r.input_power)

            sk_ion_rate[sk_ion_rate == 0] = np.nan

            ax.plot(T_e, sk_ion_rate, '--', color=c)

        if compare_sd1d:
            sd1d_ion_rate = SD1D_ion_rate()
            ax.plot(sd1d_ion_rate[:, 0], sd1d_ion_rate[:,
                                                       1], '--', color='black', label='SD1D')

        compare_adas = False
        if compare_adas:
            adas_ion_rate = ADAS_ion_rate()
            ax.plot(adas_ion_rate[:10, 0],
                    adas_ion_rate[:10, 1], '.-', label='ADAS')

        legend_lines.append(Line2D([0], [0], linestyle='-', color='black'))
        legend_lines.append(Line2D([0], [0], linestyle='--', color='black'))
        legend_labels.append(self.label1)
        legend_labels.append(self.label2)

        ax.legend(legend_lines, legend_labels)

        ax.set_xlabel('$T_e$ [eV]')
        ax.set_ylabel(r'$\langle \sigma v \rangle^{ion}$ [m$^3$ s$^{-1}$]')
        ax.set_title('Ionisation rate')
        ax.grid()
        # plt.xscale('log')
        plt.yscale('log')

    def ex_E_rate(self, compare_sd1d=True, sort_by='density'):

        if sort_by == 'power':
            self.rundeck1.sort_by_power()
            self.rundeck2.sort_by_power()
        elif sort_by == 'density':
            self.rundeck1.sort_by_density()
            self.rundeck2.sort_by_density()

        fig, ax = plt.subplots(1)
        cmap = plt.cm.get_cmap('plasma')
        if sort_by == 'power':
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
                vmin=self.rundeck1.runs[0].input_power, vmax=self.rundeck1.runs[-1].input_power))
        elif sort_by == 'density':
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
                vmin=self.rundeck1.runs[0].avg_density, vmax=self.rundeck1.runs[-1].avg_density))

        legend_lines = []
        legend_labels = []
        for r in self.rundeck1.runs:
            r.load_vars(['EX_E_RATE', 'DEEX_E_RATE',
                         'ION_E_RATE', 'REC_3B_E_RATE'])
            # q_n = (r.T_norm * r.n_norm * r.v_th / r.x_norm) * \
            #     (r.data['EX_E_RATE'] + r.data['ION_E_RATE'] -
            #     r.data['DEEX_E_RATE'] - r.data['REC_3B_E_RATE'])
            q_n = (r.T_norm * r.n_norm * r.v_th / r.x_norm) * \
                (r.data['EX_E_RATE'] - r.data['DEEX_E_RATE'])
            try:
                n_n = np.sum(r.data['NEUTRAL_DENS'], 1) * r.n_norm
            except(np.AxisError):
                n_n = r.data['NEUTRAL_DENS'] * r.n_norm

            n_e = r.data['DENSITY'] * r.n_norm
            T_e = r.data['TEMPERATURE'] * r.T_norm

            sigma_v_eps = q_n / (n_n * n_e)

            if sort_by == 'density':
                profile_label = r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                    r.avg_density/1e19) + r'$\times 10^{19}$m$^{-3}$'
                c = sm.to_rgba(r.avg_density)
            elif sort_by == 'power':
                profile_label = r'$P_{in}$' + ' = {:1.1f}'.format(
                    r.input_power) + r'MWm$^{-2}$'
                c = sm.to_rgba(r.input_power)

            if sort_by == 'power':
                legend_lines.append(
                    Line2D([0], [0], color=sm.to_rgba(r.input_power)))
            elif sort_by == 'density':
                legend_lines.append(
                    Line2D([0], [0], color=sm.to_rgba(r.avg_density)))
            legend_labels.append(profile_label)

            sigma_v_eps[sigma_v_eps == 0] = np.nan

            ax.plot(T_e, sigma_v_eps, label=profile_label, color=c)

        for r in self.rundeck2.runs:
            r.load_vars(['EX_E_RATE', 'DEEX_E_RATE',
                         'ION_E_RATE', 'REC_3B_E_RATE'])
            # q_n = (r.T_norm * r.n_norm * r.v_th / r.x_norm) * \
            #     (r.data['EX_E_RATE'] + r.data['ION_E_RATE'] -
            #     r.data['DEEX_E_RATE'] - r.data['REC_3B_E_RATE'])
            q_n = (r.T_norm * r.n_norm * r.v_th / r.x_norm) * \
                (r.data['EX_E_RATE'] - r.data['DEEX_E_RATE'])
            try:
                n_n = np.sum(r.data['NEUTRAL_DENS'], 1) * r.n_norm
            except(np.AxisError):
                n_n = r.data['NEUTRAL_DENS'] * r.n_norm

            n_e = r.data['DENSITY'] * r.n_norm
            T_e = r.data['TEMPERATURE'] * r.T_norm

            sigma_v_eps = q_n / (n_n * n_e)

            if sort_by == 'density':
                profile_label = r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                    r.avg_density/1e19) + r'$\times 10^{19}$m$^{-3}$'
                c = sm.to_rgba(r.avg_density)
            elif sort_by == 'power':
                profile_label = r'$P_{in}$' + ' = {:1.1f}'.format(
                    r.input_power) + r'MWm$^{-2}$'
                c = sm.to_rgba(r.input_power)

            sigma_v_eps[sigma_v_eps == 0] = np.nan

            ax.plot(T_e, sigma_v_eps, '--', color=c)

        if compare_sd1d:
            sd1d_ion_rate = SD1D_ion_rate()
            ax.plot(sd1d_ion_rate[:, 0], sd1d_ion_rate[:,
                                                       1], '--', color='black', label='SD1D')

        compare_adas = False
        if compare_adas:
            adas_ion_rate = ADAS_ion_rate()
            ax.plot(adas_ion_rate[:10, 0],
                    adas_ion_rate[:10, 1], '.-', label='ADAS')

        legend_lines.append(Line2D([0], [0], linestyle='-', color='black'))
        legend_lines.append(Line2D([0], [0], linestyle='--', color='black'))
        legend_labels.append(self.label1)
        legend_labels.append(self.label2)

        ax.legend(legend_lines, legend_labels)

        ax.set_xlabel('$T_e$ [eV]')
        ax.set_ylabel(
            r'$\langle \sigma v \varepsilon \rangle^{ex}$ [eVm$^3$ s$^{-1}$]')
        ax.set_title('Excitation+deexcitation energy loss rate')
        ax.grid()
        # plt.xscale('log')
        plt.yscale('log')


class SKRundeck:
    def __init__(self, rundeck_dir, sort_by='density', identifier='Output_'):
        self.dir = rundeck_dir
        self.load_rundeck(identifier)
        if sort_by == 'density':
            self.sort_by_density()
        elif sort_by == 'power':
            self.sort_by_power()
        elif sort_by == 'num_neuts':
            self.sort_by_num_neuts()
        self.sort_by = sort_by
        self.upstream_densities, self.avg_densities = self.get_densities()
        self.input_el_powers, self.input_ion_powers = self.get_input_powers()
        self.num_sims = len(self.runs)
        self.get_line_colours()
        self.get_line_labels()

    def load_rundeck(self, identifier):
        simulation_folders = [r for r in os.listdir(
            self.dir) if identifier in r and os.path.isdir(os.path.join(self.dir, r))]
        self.runs = []
        for s in simulation_folders:
            run_numbers = sorted([rn for rn in os.listdir(
                os.path.join(self.dir, s)) if 'Run_' in rn], key=lambda x: int(x.strip('Run_')))
            run_number = run_numbers[-1]
            self.runs.append(SKRun(os.path.join(self.dir, s, run_number)))

    def get_densities(self):
        upstream_densities = [None] * len(self.runs)
        avg_densities = [None] * len(self.runs)
        for i, run in enumerate(self.runs):
            upstream_densities[i] = run.upstream_density
            avg_densities[i] = run.avg_density
        return upstream_densities, avg_densities

    def get_input_powers(self):
        el_powers = [None] * len(self.runs)
        ion_powers = [None] * len(self.runs)
        for i, run in enumerate(self.runs):
            el_powers[i], ion_powers[i] = run.get_input_power()
        return el_powers, ion_powers

    def get_target_conditions(self):
        target_fluxes = [None] * len(self.runs)
        target_temps = [None] * len(self.runs)
        for i, run in enumerate(self.runs):
            target_fluxes[i], target_temps[i] = run.get_target_conditions()
        return target_fluxes, target_temps

    def part_source_plot(self):
        self.sort_by_density()
        fig, ax = plt.subplots(1)

        for i, r in enumerate(self.runs):

            profile_label = r'$n_{u}$' + ' = {:1.1f}'.format(
                r.upstream_density/1e19) + r'$\times 10^{19}$m$^{-3}$'

            r.load_vars(['S_ION_SK', 'S_ION_M'])
            S_diff = 100 * (r.data['S_ION_SK'] - r.data['S_ION_M']
                            ) / r.data['S_ION_SK']
            ax.plot(r.xgrid[::2], S_diff[::2],
                    label=profile_label, color=self.line_colours[i])

            ax.set_xlabel('x [m]')
            ax.set_ylabel(
                r'$(S_{ion}^{kin} - S_{ion}^{Max}) / S_{ion}^{kin}$ [%]')

    def get_line_colours(self):

        cmap = plt.cm.get_cmap('plasma')
        if self.sort_by == 'density':
            self.sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
                vmin=self.avg_densities[0], vmax=self.avg_densities[-1]))
        elif self.sort_by == 'power':
            self.sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
                vmin=self.input_el_powers[0], vmax=self.input_el_powers[-1]))

        self.line_colours = [None] * self.num_sims
        for i in range(self.num_sims):
            if self.sort_by == 'density':
                self.line_colours[i] = self.sm.to_rgba(self.avg_densities[i])
            elif self.sort_by == 'power':
                self.line_colours[i] = self.sm.to_rgba(self.input_el_powers[i])

    def get_line_labels(self):

        self.line_labels = [None] * self.num_sims
        for i in range(self.num_sims):
            if self.sort_by == 'density':
                if i == 0:
                    self.line_labels[i] = r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                        self.avg_densities[i]/1e19) + r'$\times 10^{19}$m$^{-3}$'
                else:
                    self.line_labels[i] = r'{:1.1f}'.format(
                        self.avg_densities[i]/1e19) + r'$\times 10^{19}$m$^{-3}$'
            elif self.sort_by == 'power':
                if i == 0:
                    self.line_labels[i] = r'$P_{in}$' + ' = {:1.1f}'.format(
                        self.input_el_powers[i]) + r'MWm$^{-2}$'
                else:
                    self.line_labels[i] = r'{:1.1f}'.format(
                        self.input_el_powers[i]) + r'MWm$^{-2}$'

    def profile(self, variable, sort_by='density', cells=(0, None)):

        fig, ax = plt.subplots(1)

        for i, r in enumerate(self.runs):
            norm_value = r.get_var_norm(variable)
            y_label = r.get_var_label(variable)
            r.load_vars(variable)
            if variable == 'NEUTRAL_DENS':
                if r.num_neutrals > 1:
                    plot_data = np.sum(r.data[variable], 1) * norm_value
                else:
                    plot_data = r.data[variable] * norm_value
            else:
                plot_data = r.data[variable] * norm_value

            ax.plot(r.xgrid[cells[0]:cells[1]:2], plot_data[cells[0]:cells[1]:2],
                    label=self.line_labels[i], color=self.line_colours[i])

        ax.legend()
        ax.set_ylabel(y_label)
        ax.set_xlabel('x [m]')
        # fig.tight_layout()
        plt.grid()

    def temp_anisotropy_plot(self, sort_by='density', cells=(0, None)):

        fig, ax = plt.subplots(1)

        for i, r in enumerate(self.runs):
            y_label = 'Percentage electron pressure anisotropy [($P_{\perp} - P_{\parallel}) / P$]'
            r.load_vars(['TEMPERATURE_PAR', 'TEMPERATURE_PERP'])
            anis = 100 * (r.data['TEMPERATURE_PERP'] -
                          r.data['TEMPERATURE_PAR']) / r.data['TEMPERATURE']
            ax.plot(r.xgrid[cells[0]:cells[1]], anis[cells[0]:cells[1]],
                    label=self.line_labels[i], color=self.line_colours[i])

        ax.legend()
        ax.set_ylabel(y_label)
        ax.set_xlabel('x [m]')
        fig.tight_layout()

    def ionization_cost_plot(self, sort_by='density'):

        self.target_fluxes, self.target_temps = self.get_target_conditions()
        E_iz = np.zeros(len(self.runs))

        for i, r in enumerate(self.runs):
            input_powers, q_E, q_sh_e, q_sh_i, q_sh_n, q_recyc_n, q_n_e, q_ion_i, q_rec_i, q_rad, q_ie, q_cx = r.get_energy_channels()
            E_iz[i] = (1e6 / el_charge) * (q_n_e + q_rad) / \
                self.target_fluxes[i]

        fig, ax = plt.subplots()
        ax.plot(self.target_temps, E_iz, 'x-', markersize=2.0)
        ax.set_xlabel('$T_{e,t}$ [eV]')
        ax.set_ylabel('$E_{iz}$ [eV]')

    def flux_rollover_plot(self):
        if self.sort_by == 'density':
            self.upstream_densities, self.avg_densities = self.get_densities()
            self.target_fluxes, self.target_temps = self.get_target_conditions()

            fig, ax = plt.subplots(1)
            ax.plot(self.upstream_densities,
                    self.target_fluxes, 'x-', color='black')
            ax2 = ax.twinx()
            ax2.plot(self.upstream_densities,
                     self.target_temps, 'x-', color='red')
            ax.set_ylabel('Target ion flux [m$^{-2}$s$^{-1}$]')
            ax2.set_ylabel('Target electron temperature [eV]', color='red')
            ax2.tick_params(axis='y', colors='red')
            ax.set_xlabel(r'Upstream density [m$^{-3}$]')
            fig.tight_layout()
        elif self.sort_by == 'power':
            self.target_fluxes, self.target_temps = self.get_target_conditions()

            fig, ax = plt.subplots(1)
            ax.plot(self.input_powers, self.target_fluxes, 'x-', color='black')
            ax2 = ax.twinx()
            ax2.plot(self.input_powers, self.target_temps, 'x-', color='red')
            ax.set_ylabel('Target ion flux [a.u.]')
            ax2.set_ylabel('Target electron temperature [eV]', color='red')
            ax2.tick_params(axis='y', colors='red')
            ax.set_xlabel(r'Input power [MWm$^{-2}$]')
            fig.tight_layout()

    def pressure_ratio_plot(self):
        if self.sort_by == 'density':

            upstream_pressure = np.zeros(len(self.runs))
            target_pressure = np.zeros(len(self.runs))
            self.upstream_densities, self.avg_densities = self.get_densities()
            for i, r in enumerate(self.runs):
                r.calc_pressure()
                plasma_pressure = r.el_pressure_stat + r.el_pressure_dyn + \
                    r.ion_pressure_stat + r.ion_pressure_dyn
                # plasma_pressure = r.el_pressure_stat + \
                #     r.ion_pressure_stat
                upstream_pressure[i] = plasma_pressure[0]
                target_pressure[i] = plasma_pressure[-1]
            fig, ax = plt.subplots()
            ax.plot(self.upstream_densities,
                    target_pressure/upstream_pressure, 'x-')
            ax.set_xlabel('Upstream density [m$^{-3}$]')
            ax.set_ylabel('Total plasma pressure ratio ($P_t/P_u$)')

    def energy_discrepancy_plot(self):

        if self.runs[0].neut_mom_eqn is False and self.runs[0].ion_temp_eqn is False:
            discrepancies = np.zeros(len(self.runs))
            input_powers = np.zeros(len(self.runs))
            for i, run in enumerate(self.runs):
                input_powers[i], q_E, q_sh_e, q_sh_i, q_sh_n, q_recyc_n, q_n_e, q_ion_i, q_rec_i, q_rad, q_ie, q_cx = run.get_energy_channels()
                discrepancies[i] = (
                    input_powers[i] - (q_sh_e + q_n_e + q_rad + q_E))

            fix, ax = plt.subplots(1)
            if self.sort_by == 'power':
                ax.plot(input_powers, discrepancies)
                ax.set_ylabel('Discrepancy [MW$m^{-2}$]')
                ax.set_xlabel('Input power [MW$m^{-2}$]')
                ax.set_title(r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                    self.runs[0].avg_density/1e19) + r'$\times 10^{19}$m$^{-3}$', loc='left')
            elif self.sort_by == 'density':
                _, densities = self.get_densities()
                ax.plot(densities, discrepancies)
                ax.set_ylabel('Discrepancy [MW$m^{-2}$]')
                ax.set_xlabel(r'$\langle n_{heavy} \rangle$' + ' [m$^{-3}$]')
                ax.set_title(
                    '$P_{in} = $' + str(input_powers[0]) + 'MWm$^{-2}$', loc='left')
                return densities, discrepancies

        if self.runs[0].neut_mom_eqn is False and self.runs[0].ion_temp_eqn is True:
            el_discrepancies = np.zeros(len(self.runs))
            ion_discrepancies = np.zeros(len(self.runs))
            input_powers = np.zeros(len(self.runs))
            for i, run in enumerate(self.runs):
                input_powers[i], q_E, q_sh_e, q_sh_i, q_sh_n, q_recyc_n, q_n_e, q_ion_i, q_rec_i, q_rad, q_ie, q_cx = run.get_energy_channels()
                el_discrepancies[i] = input_powers[i]/2 + \
                    q_ie - (q_sh_e + q_n_e + q_rad + q_E)
                ion_discrepancies[i] = input_powers[i]/2 + \
                    q_E + q_ion_i - (q_ie + q_rec_i + q_sh_i)

            fix, ax = plt.subplots(1)
            if self.sort_by == 'power':
                ax.plot(input_powers, el_discrepancies,
                        color='red', label='electrons')
                ax.plot(input_powers, ion_discrepancies,
                        color='blue', label='ions')
                ax.plot(input_powers, el_discrepancies +
                        ion_discrepancies, color='black', label='total')
                ax.set_ylabel('Discrepancy [MW$m^{-2}$]')
                ax.set_xlabel('Input power [MW$m^{-2}$]')
                ax.set_title(r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                    self.runs[0].avg_density/1e19) + r'$\times 10^{19}$m$^{-3}$', loc='left')
                ax.legend()
            elif self.sort_by == 'density':
                _, densities = self.get_densities()
                ax.plot(densities, el_discrepancies,
                        color='red', label='electrons')
                ax.plot(densities, ion_discrepancies,
                        color='blue', label='ions')
                ax.plot(densities, el_discrepancies+ion_discrepancies,
                        color='black', label='total')
                ax.set_ylabel('Discrepancy [MW$m^{-2}$]')
                ax.set_xlabel(r'$\langle n_{heavy} \rangle$' + ' [m$^{-3}$]')
                ax.set_title(
                    '$P_{in} = $' + str(input_powers[0]) + 'MWm$^{-2}$', loc='left')
                ax.legend()
                return densities, el_discrepancies, ion_discrepancies

        if self.runs[0].neut_mom_eqn is True and self.runs[0].ion_temp_eqn is True:
            el_discrepancies = np.zeros(len(self.runs))
            ion_discrepancies = np.zeros(len(self.runs))
            neut_discrepancies = np.zeros(len(self.runs))
            input_powers = np.zeros(len(self.runs))
            for i, run in enumerate(self.runs):
                input_powers[i], q_E, q_sh_e, q_sh_i, q_sh_n, q_recyc_n, q_n_e, q_ion_i, q_rec_i, q_rad, q_ie, q_cx = run.get_energy_channels()
                el_discrepancies[i] = input_powers[i]/2 + \
                    q_ie - (q_sh_e + q_n_e + q_rad + q_E)
                ion_discrepancies[i] = input_powers[i]/2 + \
                    q_E + q_ion_i - (q_ie + q_rec_i + q_sh_i + q_cx)
                neut_discrepancies[i] = q_cx + q_rec_i + \
                    q_recyc_n - (q_ion_i + q_sh_n)

            fix, ax = plt.subplots(1)
            if self.sort_by == 'power':
                ax.plot(input_powers, el_discrepancies,
                        color='red', label='electrons')
                ax.plot(input_powers, ion_discrepancies,
                        color='blue', label='ions')
                ax.plot(input_powers, neut_discrepancies,
                        color='green', label='neutrals')
                ax.plot(input_powers, el_discrepancies+ion_discrepancies,
                        color='black', label='total plasma')
                ax.set_ylabel('Discrepancy [MW$m^{-2}$]')
                ax.set_xlabel('Input power [MW$m^{-2}$]')
                ax.set_title(r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                    self.runs[0].avg_density/1e19) + r'$\times 10^{19}$m$^{-3}$', loc='left')
                ax.legend()
            elif self.sort_by == 'density':
                _, densities = self.get_densities()
                ax.plot(densities, el_discrepancies,
                        color='red', label='electrons')
                ax.plot(densities, ion_discrepancies,
                        color='blue', label='ions')
                ax.plot(densities, neut_discrepancies,
                        color='green', label='neutrals')
                ax.plot(densities, el_discrepancies+ion_discrepancies,
                        color='black', label='total plasma')
                ax.set_ylabel('Discrepancy [MW$m^{-2}$]')
                ax.set_xlabel(r'$\langle n_{heavy} \rangle$' + ' [m$^{-3}$]')
                ax.set_title(
                    '$P_{in} = $' + str(input_powers[0]) + 'MWm$^{-2}$', loc='left')
                ax.legend()
                return densities, el_discrepancies, ion_discrepancies, neut_discrepancies

    def energy_channels_plot(self):

        input_powers = np.zeros(len(self.runs))
        q_E = np.zeros(len(self.runs))
        q_sh_e = np.zeros(len(self.runs))
        q_sh_i = np.zeros(len(self.runs))
        q_sh_n = np.zeros(len(self.runs))
        q_recyc_n = np.zeros(len(self.runs))
        q_n_e = np.zeros(len(self.runs))
        q_ion_i = np.zeros(len(self.runs))
        q_rec_i = np.zeros(len(self.runs))
        q_rad = np.zeros(len(self.runs))
        q_ie = np.zeros(len(self.runs))
        q_cx = np.zeros(len(self.runs))
        for i, run in enumerate(self.runs):
            input_powers[i], q_E[i], q_sh_e[i], q_sh_i[i], q_sh_n[i], q_recyc_n[i], q_n_e[
                i], q_ion_i[i], q_rec_i[i], q_rad[i], q_ie[i], q_cx[i] = run.get_energy_channels()

        if self.runs[0].neut_mom_eqn is False and self.runs[0].ion_temp_eqn is False:
            # Diffusive neutrals, no T_i equation
            fig, ax = plt.subplots(1)
            if self.sort_by == 'density':
                self.sort_by_density()
                ax.stackplot(self.densities, q_rad, q_n_e, q_E, q_sh_e)
                ax.set_xlabel('Upstream density [m$^{-3}$]')
            elif self.sort_by == 'power':
                ax.stackplot(input_powers, q_rad, q_n_e, q_E, q_sh_e)
                ax.set_xlabel('Input power [MWm$^{-2}$]')

            ax.legend(['Radiated power', 'Given to neutrals',
                       'E field', 'Sheath'], loc='upper left')
            ax.set_ylabel('Heat flux [MWm$^{-2}$]')

        if self.runs[0].neut_mom_eqn is False and self.runs[0].ion_temp_eqn is True:
            # Diffusive neutrals, with T_i equation

            # fig, ax = plt.subplots(3,1, figsize=(10,20))
            fig, ax = plt.subplots(1)

            if self.sort_by == 'density':
                self.sort_by_density()
                ax.stackplot(self.densities, q_rad, q_rec_i -
                             q_ion_i, q_n_e, q_sh_i, q_sh_e)
                ax.set_xlabel('Line-averaged density [m$^{-3}$]')

            if self.sort_by == 'power':
                ax.stackplot(input_powers, q_rad, q_rec_i -
                             q_ion_i, q_n_e, q_sh_i, q_sh_e)
                ax.set_xlabel('Total input power [MWm$^{-2}$]')

            ax.legend(['Radiated', 'Heavy particle transfer', 'Given to neutrals by electrons',
                       'Sheath (ions)', 'Sheath (electrons)'], loc='upper left')

            ax.set_ylabel('Heat flux [MWm$^{-2}$]')

        if self.runs[0].neut_mom_eqn is True and self.runs[0].neut_temp_eqn is True and self.runs[0].ion_temp_eqn is True:
            fig, ax = plt.subplots(1)
            # ax.plot(self.densities, q_sh_i + q_sh_e + q_rec_i - q_ion_i + q_cx + q_n_e + q_rad, '-', color='black')
            ax.plot(self.densities, input_powers, '--', color='black')
            ax.stackplot(self.densities, q_sh_i, q_sh_e, q_n_e,
                         q_rad, q_rec_i - q_ion_i + q_cx)
            ax.legend(['expected total', 'sheath (ions)', 'sheath (electrons)',
                       'given to neutrals (electrons)', 'radiation', 'CX, etc'], loc='lower left')
            ax.set_xlabel('Line-averaged density [m$^{-3}$]')
            ax.set_ylabel('Heat flux [MWm$^{-2}$]')

    def target_plot(self, variable='DENSITY'):
        if variable == 'DENSITY':
            n_e_boundary = np.zeros(len(self.runs))
            densities = np.zeros(len(self.runs))
            for i, r in enumerate(self.runs):
                densities[i] = r.upstream_density
                n_e_boundary[i] = r.get_target_density() * r.n_norm

            fig, ax = plt.subplots(1)
            ax.plot(densities, n_e_boundary)
            ax.set_xlabel('Upstream density [m$^{-3}$]')
            ax.set_ylabel('Target density [m$^{-3}$]')
        elif variable == 'TEMPERATURE':
            T_e_boundary = np.zeros(len(self.runs))
            densities = np.zeros(len(self.runs))
            for i, r in enumerate(self.runs):
                densities[i] = r.upstream_density
                T_e_boundary[i] = r.data['TEMPERATURE'][-1] * r.T_norm

            fig, ax = plt.subplots(1)
            ax.plot(densities, T_e_boundary)
            ax.set_xlabel('Upstream density [m$^{-3}$]')
            ax.set_ylabel('Target electron temperature [eV]')

    def sort_by_power(self):
        self.input_powers = [None] * len(self.runs)
        for i, run in enumerate(self.runs):
            self.input_powers[i], _ = run.get_input_power()
            run.input_power = self.input_powers[i]
        self.runs = [r for _, r in sorted(
            zip(self.input_powers, self.runs), key=lambda pair: pair[0])]

    def sort_by_density(self):
        self.densities = [None] * len(self.runs)
        for i, run in enumerate(self.runs):
            self.densities[i] = run.avg_density
        self.runs = [r for _, r in sorted(
            zip(self.densities, self.runs), key=lambda pair: pair[0])]

    def sort_by_num_neuts(self):
        self.num_neutrals = [None] * len(self.runs)
        for i, run in enumerate(self.runs):
            self.num_neutrals[i] = run.num_neutrals
        self.runs = [r for _, r in sorted(
            zip(self.num_neutrals, self.runs), key=lambda pair: pair[0])]

    def pressure_plots(self, sort_by='density', option=1, interp=False, cells=(0, -1)):
        if sort_by == 'density':
            self.sort_by_density()
        if sort_by == 'power':
            self.sort_by_power()
        for r in self.runs:
            r.pressure_plot(option=option, interp=interp, cells=cells)

    def sankey_diagrams(self, sort_by='density', separate=True):
        if sort_by == 'density':
            self.sort_by_density()
        if sort_by == 'power':
            self.sort_by_power()
        for r in self.runs:
            r.sankey_diagram(separate=separate)

    def ion_rate(self, compare_sd1d=True, sort_by='density'):

        if sort_by == 'power':
            self.sort_by_power()
        elif sort_by == 'density':
            self.sort_by_density()

        fig, ax = plt.subplots(1)

        sk_ion_rate = []
        for i, r in enumerate(self.runs):
            r.load_vars(['S_ION_SK'])

            S_ion = r.n_norm * (r.v_th / r.x_norm) * r.data['S_ION_SK']
            try:
                n_n = np.sum(r.data['NEUTRAL_DENS'], 1) * r.n_norm
            except(np.AxisError):
                n_n = r.data['NEUTRAL_DENS'] * r.n_norm

            n_e = r.data['DENSITY'] * r.n_norm

            T_e = r.data['TEMPERATURE'] * r.T_norm

            sk_ion_rate = (S_ion / (n_n * n_e))

            sk_ion_rate[sk_ion_rate == 0] = np.nan

            ax.plot(T_e, sk_ion_rate,
                    label=self.line_labels[i], color=self.line_colours[i])

        if compare_sd1d:
            sd1d_ion_rate = SD1D_ion_rate()
            ax.plot(sd1d_ion_rate[:, 0], sd1d_ion_rate[:,
                                                       1], '--', color='black', label='SD1D')

        compare_adas = False
        if compare_adas:
            adas_ion_rate = ADAS_ion_rate()
            ax.plot(adas_ion_rate[:10, 0],
                    adas_ion_rate[:10, 1], '.-', label='ADAS')

        ax.set_xlabel('$T_e$ [eV]')
        ax.set_ylabel(r'$\langle \sigma v \rangle$ [m$^3$ s$^{-1}$]')
        ax.legend()
        ax.grid()
        plt.xscale('log')
        plt.yscale('log')


class SKRun:
    def __init__(self, run_dir=None):
        if run_dir is None:
            self.dir = '/Users/dpower/Documents/01 - PhD/01 - Code/01 - SOL-KiT'
        else:
            run_dir_list = os.listdir(run_dir)
            if 'INPUT' in run_dir_list:
                self.dir = run_dir
            else:
                run_dir_list = sorted([d for d in run_dir_list if 'Run_' in d])
                if run_dir_list == []:
                    print('This does not look like a valid SKRun directory.')
                else:
                    self.dir = os.path.join(run_dir, run_dir_list[-1])

        self.load_all()

    def load_all(self):

        self.load_norms()
        self.load_grids()
        self.load_switches()
        self.load_neut_data()
        _, _ = self.get_input_power()

        # Load output data
        variables = ['TEMPERATURE', 'DENSITY', 'ION_DENS',
                     'FLOW_VEL_X', 'NEUTRAL_DENS', 'ION_VEL']
        # variables = ['TEMPERATURE', 'DENSITY', 'ION_DENS',
        #              'FLOW_VEL_X', 'ION_VEL']
        if self.ion_temp_eqn:
            variables.append('ION_TEMPERATURE')
        if self.neut_mom_eqn:
            variables.append('NEUTRAL_VEL')
            if self.twod_neuts:
                variables.append('NEUTRAL_VEL_PERP')
        if self.neut_temp_eqn:
            variables.append('NEUTRAL_TEMPERATURE')
        self.data = {}
        self.load_vars(variables)

        self.upstream_density = self.data['DENSITY'][0]*self.n_norm
        self.load_avg_density()

    def load_avg_density(self):
        self.connection_length = self.xgrid[-1] + \
            self.dxc[-1]/2 + self.dxc[0]/2
        self.plasma_density = self.integrate_x(
            self.data['DENSITY']*self.n_norm) / self.connection_length
        if self.num_neutrals > 1:
            try:
                self.neutral_density = self.integrate_x(
                    np.sum(self.data['NEUTRAL_DENS'], 1)*self.n_norm) / self.connection_length
            except:
                print('Only 1 neutral state found in output.')
                try:
                    self.neutral_density = self.integrate_x(
                        self.data['NEUTRAL_DENS']*self.n_norm) / self.connection_length
                except:
                    self.neutral_density = 0.0
        else:
            self.neutral_density = self.integrate_x(
                self.data['NEUTRAL_DENS']*self.n_norm) / self.connection_length
        self.avg_density = self.plasma_density + self.neutral_density

    def load_vars(self, variables):
        if type(variables) == list:
            for variable in variables:
                if variable not in self.data.keys():
                    try:
                        var_dir = os.path.join(self.dir, 'OUTPUT', variable)
                        var_files = [vf for vf in sorted(
                            os.listdir(var_dir)) if variable in vf]
                        self.data[variable] = np.loadtxt(
                            os.path.join(var_dir, var_files[-1]))
                    except(FileNotFoundError):
                        print('WARNING: Variable ' +
                              variable + ' not found in output')
        elif type(variables) == str:
            variable = variables
            if variable not in self.data.keys():
                try:
                    var_dir = os.path.join(self.dir, 'OUTPUT', variable)
                    var_files = sorted(os.listdir(var_dir))
                    self.data[variable] = np.loadtxt(
                        os.path.join(var_dir, var_files[-1]))
                except(FileNotFoundError):
                    print('WARNING: Variable ' +
                          variable + ' not found in output')

    def load_dist(self):
        var_dir = os.path.join(self.dir, 'OUTPUT', 'DIST_F')
        var_files = sorted(os.listdir(var_dir))
        self.data['DIST_F'] = [None] * (self.l_max + 1)
        for l in range(self.l_max+1):
            # print(var_files[-11:-9], str(l) + '_')
            l_files = sorted([
                f for f in var_files if f[-11:-9] == str(l) + '_'])
            self.data['DIST_F'][l] = np.loadtxt(
                os.path.join(var_dir, l_files[-1]))

    def load_neut_data(self):
        neut_file = os.path.join(self.dir, 'INPUT', 'NEUT_AND_HEAT_INPUT.txt')
        with open(neut_file) as f:
            l = f.readlines()
            for line in l:
                if 'NUM_NEUTRALS' in line:
                    self.num_neutrals = int(line.split()[2])
                if 'REC_R' in line:
                    self.rec_r = float(line.split()[2])
                if 'PITCH_ANGLE' in line:
                    self.pitch_angle = float(line.split()[2])

    def load_switches(self):
        switches_file = os.path.join(self.dir, 'INPUT', 'SWITCHES_INPUT.txt')
        with open(switches_file) as f:
            l = f.readlines()
            self.ion_temp_eqn = False
            self.neut_temp_eqn = False
            self.neut_mom_eqn = False
            self.twod_neuts = False
            self.intrinsic_coupling = False
            self.full_fluid = False
            self.periodic = False
            for line in l:
                if 'ION_TEMP_EQN_SWITCH' in line:
                    if line.split()[2] == 'T':
                        self.ion_temp_eqn = True
                if 'NEUTRAL_TEMP_SWITCH' in line:
                    if line.split()[2] == 'T':
                        self.neut_temp_eqn = True
                if 'NEUTRAL_MOMENTUM_SWITCH' in line:
                    if line.split()[2] == 'T':
                        self.neut_mom_eqn = True
                if 'MIMIC_2D_NEUTRALS' in line:
                    if line.split()[2] == 'T':
                        self.twod_neuts = True
                if 'FULL_FLUID_MODE' in line:
                    if line.split()[2] == 'T':
                        self.full_fluid = True
                if 'PERIODIC_BOUNDARY_SWITCH' in line:
                    if line.split()[2] == 'T':
                        self.periodic = True
                if 'INTRINSIC_COUPLING_SWITCH' in line:
                    if line.split()[2] == 'T':
                        self.intrinsic_coupling = True

    def load_norms(self):

        with open(os.path.join(self.dir, 'INPUT', 'GRID_INPUT.txt')) as f:
            grid_input_data = f.readlines()
            num_t = int(grid_input_data[1].split()[2])
            dt = float(grid_input_data[3].split()[2])
            N_x = int(grid_input_data[16].split()[2])
            dx = float(grid_input_data[17].split()[2])

        with open(os.path.join(self.dir, 'INPUT', 'NORMALIZATION_INPUT.txt')) as f:
            norm_input_data = f.readlines()
            Z = float(norm_input_data[0].split()[2])
            T_norm = float(norm_input_data[2].split()[2])
            n_norm = float(norm_input_data[3].split()[2])

        with open(os.path.join(self.dir, 'INPUT', 'F_INIT_INPUT.txt')) as f:
            f_init_data = f.readlines()
            T_0_init = float(f_init_data[1].split()[2])

        gamma_ee_0 = el_charge ** 4 / (4 * np.pi * (el_mass * epsilon_0) ** 2)
        T_norm_J = T_norm * el_charge
        gamma_ei_0 = Z ** 2 * gamma_ee_0

        self.v_th = np.sqrt(2.0 * T_norm_J / el_mass)
        self.t_norm = self.v_th ** 3 / \
            (gamma_ei_0 * n_norm * lambda_ei(1.0, 1.0, T_norm, n_norm, Z)/Z)
        self.x_norm = self.v_th * self.t_norm
        self.T_norm = T_norm
        self.n_norm = n_norm
        self.sigma_0 = np.pi * bohr_radius ** 2
        self.dt = dt

    def load_grids(self):

        # Load x grid
        x_grid_file = os.path.join(self.dir, 'OUTPUT', 'GRIDS', 'X_GRID.txt')
        with open(x_grid_file) as f:
            x_grid = f.readlines()
            x_grid = [float(x) for x in x_grid]
        # x_grid = np.array([x + x_grid[1] for x in x_grid]) * self.x_norm
        x_grid = np.array(x_grid) * self.x_norm

        dxc = [None] * len(x_grid)
        dxc[0] = 2 * (x_grid[1] - x_grid[0])
        dxc[-1] = 2 * (x_grid[-1] - x_grid[-2])
        for i in range(1, len(x_grid)-1):
            dxc[i] = x_grid[i+1] - x_grid[i-1]
        dxc = np.array(dxc)

        self.num_x = len(x_grid)
        self.xgrid = x_grid
        self.dxc = dxc

        # Load v grid
        v_grid_file = os.path.join(self.dir, 'OUTPUT', 'GRIDS', 'V_GRID.txt')
        with open(v_grid_file) as f:
            v_grid = f.readlines()
            v_grid = [float(v) for v in v_grid]
        v_grid = np.array(v_grid) * self.v_th

        v_grid_width_file = os.path.join(
            self.dir, 'OUTPUT', 'GRIDS', 'V_GRID_WIDTH.txt')
        with open(v_grid_width_file) as f:
            dvc = f.readlines()
            dvc = [float(dv) for dv in dvc]
        dvc = np.array(dvc) * self.v_th

        self.num_v = len(v_grid)
        self.vgrid = v_grid
        self.dvc = dvc

        # Load l_max
        grid_file = os.path.join(self.dir, 'INPUT', 'GRID_INPUT.txt')
        with open(grid_file) as f:
            l = f.readlines()
            for line in l:
                if 'L_MAX' in line:
                    self.l_max = int(line.split()[2])

    def calc_pressure(self, interp=False):

        self.el_pressure_stat = el_charge * self.T_norm * \
            self.n_norm * self.data['TEMPERATURE'] * self.data['DENSITY']
        self.el_pressure_dyn = 2.0 * el_charge * self.T_norm * self.n_norm * \
            self.data['DENSITY'] * (self.data['FLOW_VEL_X'] ** 2)

        if self.ion_temp_eqn:
            self.ion_pressure_stat = el_charge * self.T_norm * self.n_norm * \
                self.data['ION_TEMPERATURE'] * self.data['ION_DENS']
            self.ion_pressure_dyn = 2.0 * (ion_mass / el_mass) * el_charge * self.T_norm * \
                self.n_norm * self.data['ION_DENS'] * \
                (self.data['ION_VEL'] ** 2)
        else:
            self.ion_pressure_stat = self.el_pressure_stat
            self.ion_pressure_dyn = \
                (ion_mass / el_mass) * self.el_pressure_dyn
        self.neut_pressure_stat = np.zeros(len(self.xgrid))
        if self.neut_temp_eqn:
            for i in range(len(self.xgrid)):
                if self.num_neutrals > 1:
                    self.neut_pressure_stat[i] = el_charge * self.T_norm * self.n_norm * np.sum(
                        self.data['NEUTRAL_TEMPERATURE'][i] * self.data['NEUTRAL_DENS'][i])
                else:
                    self.neut_pressure_stat[i] = el_charge * self.T_norm * self.n_norm * \
                        self.data['NEUTRAL_TEMPERATURE'][i] * \
                        self.data['NEUTRAL_DENS'][i]
        else:
            for i in range(len(self.xgrid)):
                if self.num_neutrals > 1:
                    self.neut_pressure_stat[i] = el_charge * self.T_norm * self.n_norm * (
                        3.0 / self.T_norm) * np.sum(self.data['NEUTRAL_DENS'][i])
                else:
                    self.neut_pressure_stat[i] = el_charge * self.T_norm * self.n_norm * (
                        3.0 / self.T_norm) * self.data['NEUTRAL_DENS'][i]
        if self.neut_mom_eqn:
            self.neut_pressure_dyn = np.zeros(len(self.xgrid))
            if self.twod_neuts:
                if self.num_neutrals > 1:
                    for i in range(len(self.xgrid)):
                        self.neut_pressure_dyn[i] = 2.0 * (ion_mass / el_mass) * el_charge * self.T_norm * self.n_norm * np.sum(self.data['NEUTRAL_DENS'][i]) * (
                            (self.data['NEUTRAL_VEL'][i] + (self.data['NEUTRAL_VEL_PERP'][i]/np.tan(np.pi*self.pitch_angle/180.0))) ** 2)
                else:
                    for i in range(len(self.xgrid)):
                        self.neut_pressure_dyn[i] = 2.0 * (ion_mass / el_mass) * el_charge * self.T_norm * self.n_norm * self.data['NEUTRAL_DENS'][i] * (
                            (self.data['NEUTRAL_VEL'][i] + (self.data['NEUTRAL_VEL_PERP'][i]/np.tan(np.pi*self.pitch_angle/180.0))) ** 2)
            else:
                if self.num_neutrals > 1:
                    for i in range(len(self.xgrid)):
                        self.neut_pressure_dyn[i] = 2.0 * (ion_mass / el_mass) * el_charge * self.T_norm * self.n_norm * np.sum(
                            self.data['NEUTRAL_DENS'][i]) * (self.data['NEUTRAL_VEL'][i] ** 2)
                else:
                    for i in range(len(self.xgrid)):
                        self.neut_pressure_dyn[i] = 2.0 * (ion_mass / el_mass) * el_charge * self.T_norm * \
                            self.n_norm * \
                            self.data['NEUTRAL_DENS'][i] * \
                            (self.data['NEUTRAL_VEL'][i] ** 2)
        else:
            self.neut_pressure_dyn = np.zeros(self.num_x)

        if interp is True:
            # Interpolate to boundaries/centres
            self.el_pressure_stat = self.interp_var(
                self.el_pressure_stat, 'centre')
            self.ion_pressure_stat = self.interp_var(
                self.ion_pressure_stat, 'centre')
            self.neut_pressure_stat = self.interp_var(
                self.neut_pressure_stat, 'centre')
            self.el_pressure_dyn = self.interp_var(
                self.el_pressure_dyn, 'centre')
            self.ion_pressure_dyn = self.interp_var(
                self.ion_pressure_dyn, 'centre')
            self.neut_pressure_dyn = self.interp_var(
                self.neut_pressure_dyn, 'centre')

    def interp_var(self, var_data, position):
        interp_var = np.zeros(len(var_data))
        boundary_val = 0.0
        if position == 'centre':
            for i in range(len(var_data)):
                if i % 2 == 0:
                    interp_var[i] = var_data[i]
            for i in range(len(var_data)):
                if i % 2 == 1:
                    interp_var[i] = 0.5*(interp_var[i-1] + interp_var[i+1])
        elif position == 'boundary':
            for i in range(len(var_data)):
                if i % 2 == 1:
                    interp_var[i] = var_data[i]
            for i in range(1, len(var_data)-2):
                if i % 2 == 0:
                    interp_var[i] = 0.5*(interp_var[i-1] + interp_var[i+1])
            interp_var[0] = 0.5*interp_var[1]
            interp_var[-1] = 0.5*(interp_var[-2] + boundary_val)

        return interp_var

    def pressure_plot(self, option=1, interp=False, cells=(0, None)):
        self.calc_pressure(interp)
        if option == 1:
            plasma_p = self.el_pressure_stat + self.ion_pressure_stat
            plasma_dp = self.el_pressure_dyn + self.ion_pressure_dyn
            tot_p = plasma_p + plasma_dp + self.neut_pressure_dyn + self.neut_pressure_stat
            fig, ax = plt.subplots(1)
            ax.plot(self.xgrid[cells[0]:cells[1]], plasma_p[cells[0]
                    :cells[1]], '-', color='blue', label='Static plasma')
            ax.plot(self.xgrid[cells[0]:cells[1]], plasma_dp[cells[0]
                    :cells[1]], '-', color='red', label='Dynamic plasma')
            ax.plot(self.xgrid[cells[0]:cells[1]], self.neut_pressure_stat[cells[0]
                    :cells[1]], '--', color='blue', label='Static neutral')
            ax.plot(self.xgrid[cells[0]:cells[1]], self.neut_pressure_dyn[cells[0]
                    :cells[1]], '--', color='red', label='Dynamic neutral')
            ax.plot(self.xgrid[cells[0]:cells[1]], tot_p[cells[0]
                    :cells[1]], '-', color='black', label='Total')
            ax.set_title(r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                self.avg_density/1e19) + r'$\times 10^{19}$m$^{-3}$')
            ax.legend()
        if option == 2:
            el_p = self.el_pressure_stat + self.el_pressure_dyn
            ion_p = self.ion_pressure_stat + self.ion_pressure_dyn
            neut_p = self.neut_pressure_stat + self.neut_pressure_dyn
            tot_p = el_p + ion_p + neut_p
            fig, ax = plt.subplots(1)
            ax.plot(self.xgrid[cells[0]:cells[1]], el_p[cells[0]
                    :cells[1]], '-', color='red', label='Electrons')
            ax.plot(self.xgrid[cells[0]:cells[1]], ion_p[cells[0]                    :cells[1]], '-', color='blue', label='Ions')
            ax.plot(self.xgrid[cells[0]:cells[1]], neut_p[cells[0]                    :cells[1]], '-', color='green', label='Neutrals')
            ax.plot(self.xgrid[cells[0]:cells[1]], tot_p[cells[0]                    :cells[1]], '-', color='black', label='Total')
            ax.set_title(r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                self.avg_density/1e19) + r'$\times 10^{19}$m$^{-3}$')
            ax.legend()
        if option == 3:
            plasma_p = self.el_pressure_stat + self.ion_pressure_stat
            plasma_dp = self.el_pressure_dyn + self.ion_pressure_dyn
            tot_p = plasma_p + plasma_dp + self.neut_pressure_dyn + self.neut_pressure_stat
            fig, ax = plt.subplots(1)
            ax.plot(self.xgrid[cells[0]:cells[1]], self.el_pressure_stat[cells[0]
                    :cells[1]], '-', color='red', label='Electrons')
            ax.plot(self.xgrid[cells[0]:cells[1]],
                    self.el_pressure_dyn[cells[0]:cells[1]], '--', color='red')
            ax.plot(self.xgrid[cells[0]:cells[1]], self.ion_pressure_stat[cells[0]:cells[1]], '-', color='blue', label='Ions')
            ax.plot(self.xgrid[cells[0]:cells[1]],
                    self.ion_pressure_dyn[cells[0]:cells[1]], '--', color='blue')
            ax.plot(self.xgrid[cells[0]:cells[1]], self.neut_pressure_stat[cells[0]
                    :cells[1]], '-', color='green', label='Neutrals')
            ax.plot(self.xgrid[cells[0]:cells[1]],
                    self.neut_pressure_dyn[cells[0]:cells[1]], '--', color='green')
            ax.plot(self.xgrid[cells[0]:cells[1]], tot_p[cells[0]                    :cells[1]], '-', color='black', label='Total')
            ax.set_title(r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                self.avg_density/1e19) + r'$\times 10^{19}$m$^{-3}$')
            ax.legend()
        if option == 4:
            plasma_p = self.el_pressure_stat + self.ion_pressure_stat + \
                self.el_pressure_dyn + self.ion_pressure_dyn
            neut_p = self.neut_pressure_dyn + self.neut_pressure_stat
            tot_p = plasma_p + neut_p
            fig, ax = plt.subplots(1)
            ax.plot(self.xgrid[cells[0]:cells[1]], plasma_p[cells[0]
                    :cells[1]], '-', color='red', label='Plasma')
            ax.plot(self.xgrid[cells[0]:cells[1]], neut_p[cells[0]:cells[1]], '-', color='green', label='Neutrals')
            ax.plot(self.xgrid[cells[0]:cells[1]], tot_p[cells[0]                    :cells[1]], '-', color='black', label='Total')
            ax.set_title(r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                self.avg_density/1e19) + r'$\times 10^{19}$m$^{-3}$')
            ax.legend()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('Pressure [Pa]')

    def get_var_label(self, variable):
        if variable == 'TEMPERATURE':
            label = '$T_e$ [eV]'
        elif variable == 'ION_TEMPERATURE':
            label = '$T_i$ [eV]'
        elif variable == 'NEUTRAL_TEMPERATURE':
            label = '$T_n$ [eV]'
        elif variable == 'ION_VEL':
            label = '$u_i$ [ms$^{-1}$]'
        elif variable == 'FLOW_VEL_X':
            label = '$u_e$ [ms$^{-1}$]'
        elif variable == 'NEUTRAL_VEL':
            label = '$u_{n\parallel}$ [ms$^{-1}$]'
        elif variable == 'NEUTRAL_VEL_PERP':
            label = '$u_{n\perp}$ [ms$^{-1}$]'
        elif variable == 'DENSITY':
            label = '$n_e$ [m$^{-3}$]'
        elif variable == 'ION_DENS':
            label = '$n_i$ [m$^{-3}$]'
        elif variable == 'NEUTRAL_DENS':
            label = '$n_n$ [m$^{-3}$]'
        elif variable == 'S_REC':
            label = 'Recombination source [m$^{-3}$s$^{-1}$]'
        elif variable == 'SH_q_RATIO':
            label = 'Heat flux ratio $(q^{Kin}/q^{SH})$'
        else:
            label = variable

        return label

    def get_var_norm(self, variable):
        if 'DENS' in variable:
            var_norm = self.n_norm
        elif 'VEL' in variable:
            var_norm = self.v_th
        elif 'TEMPERATURE' in variable:
            var_norm = self.T_norm
        elif variable == 'S_REC':
            var_norm = self.n_norm / self.t_norm
        elif 'S_ION_' in variable:
            var_norm = self.n_norm / self.t_norm
        elif 'DIST_F' in variable:
            var_norm = self.n_norm / (self.v_th ** 3)
        else:
            print("WARNING: Could not find normalisation for variable " + variable)
            var_norm = 1.0
        return var_norm

    def profile(self, variable, cells=(0, None)):

        fig, ax = plt.subplots(1)

        ylabel = self.get_var_label(variable)
        norm_value = self.get_var_norm(variable)
        if variable not in self.data.keys():
            self.load_vars(variable)
        plot_data = self.data[variable] * norm_value

        ax.plot(self.xgrid[cells[0]:cells[1]], plot_data[cells[0]:cells[1]])
        ax.set_xlabel('x [m]')
        ax.set_ylabel(ylabel)

    def get_target_density(self):
        self.target_density = self.data['DENSITY'][-1] * \
            self.data['DENSITY'][-1] / self.data['DENSITY'][-2]
        # self.target_density = 0.5 * (self.data['DENSITY'][-2] + (self.data['DENSITY'][-2] + (self.data['DENSITY'][-2] - self.data['DENSITY'][-4])/ (self.dxc[-2]) * self.dxc[-1]))
        return self.target_density

    def get_target_conditions(self):
        if self.ion_temp_eqn is True:
            T_i_boundary = self.data['ION_TEMPERATURE'][-1]
        else:
            T_i_boundary = self.data['TEMPERATURE'][-1]
        T_e_boundary = self.data['TEMPERATURE'][-1]
        n_e_boundary = self.get_target_density()
        self.n_e_boundary = n_e_boundary * self.n_norm
        flux = n_e_boundary * sound_speed(T_e_boundary, T_i_boundary)
        flux = flux * self.n_norm * self.v_th
        # target_temp = 0.5*(T_i_boundary + T_e_boundary) * self.T_norm
        target_temp = T_e_boundary * self.T_norm
        return flux, target_temp

    def integrate_x(self, int_data, on='centre'):
        # Integrate the variables provide over the spatial domain
        int_val = 0.0
        if on == 'centre':
            for i in range(self.num_x):
                if i % 2 == 0:
                    int_val = int_val + int_data[i] * self.dxc[i]
        # elif on == 'boundary':
        #     for i in range(self.num_x-1):
        #         if i % 2 == 0:
        #             int_val = int_val + int_data[i] * self.dxc[i]
        #     int_val += int_data[-1]*self.dxc[-1]*0.5
        return int_val

    def get_energy_channels(self):

        self.el_power, self.ion_power = self.get_input_power()
        if self.ion_temp_eqn:
            input_power = self.el_power + self.ion_power
        else:
            input_power = self.el_power

        # q_sh_e
        q_sh_e = 1e-6 * self.q_sh_e()

        # q_sh_i
        if self.ion_temp_eqn:
            q_sh_i = 1e-6 * self.q_sh_i()
        else:
            q_sh_i = None

        if self.neut_temp_eqn:
            q_sh_n = 1e-6*self.q_sh_n()
        else:
            q_sh_n = None

        # q_recyc_n
        if self.neut_temp_eqn:
            ion_flux, _ = self.get_target_conditions()
            T_i_boundary = self.data['ION_TEMPERATURE'][-1]
            # T_n_recyc = 3.0
            T_n_recyc = 0.5 * (T_i_boundary/2.0 + (3.0/self.T_norm))
            q_recyc_n = 1e-6 * 1.5 * el_charge * self.T_norm * T_n_recyc * \
                (ion_flux /
                 (2.0*(self.xgrid[-1] - self.xgrid[-2]))) * self.dxc[-1]
        else:
            q_recyc_n = None

        # q_E
        self.load_vars('E_FIELD_X')
        q_E = 1e-6 * (2.0 * el_charge * self.T_norm * self.n_norm / self.t_norm) * self.integrate_x(
            self.data['E_FIELD_X'] * self.data['DENSITY'] * self.data['FLOW_VEL_X'], on='centre')

        # q_n
        try:
            self.load_vars('EX_E_RATE')
            q_n_ex = self.integrate_x(self.data['EX_E_RATE'])
        except:
            q_n_ex = 0
        try:
            self.load_vars('DEEX_E_RATE')
            q_n_deex = self.integrate_x(self.data['DEEX_E_RATE'])
        except:
            q_n_deex = 0
        try:
            self.load_vars('ION_E_RATE')
            q_n_ion = self.integrate_x(self.data['ION_E_RATE'])
        except:
            q_n_ion = 0
        try:
            self.load_vars('REC_3B_E_RATE')
            q_n_rec = self.integrate_x(self.data['REC_3B_E_RATE'])
        except:
            q_n_rec = 0
        try:
            self.load_vars('RAD_DEEX_E_RATE')
            q_rad_deex = self.integrate_x(self.data['RAD_DEEX_E_RATE'])
        except:
            q_rad_deex = 0
        try:
            self.load_vars('RAD_REC_E_RATE')
            q_rad_rec = self.integrate_x(self.data['RAD_REC_E_RATE'])
        except:
            q_rad_rec = 0
        q_n_e = 1e-6 * (self.T_norm * el_charge * self.n_norm * self.v_th / self.x_norm) * \
            (-q_rad_deex - q_rad_rec + q_n_ex - q_n_deex - q_n_rec + q_n_ion)
        if self.ion_temp_eqn:
            try:
                self.load_vars(['S_ION_SK', 'S_REC'])
                q_rec_i = ion_mass/el_mass * \
                    self.integrate_x(
                        self.data['S_REC']*self.data['ION_VEL']*self.data['ION_VEL'])
                q_rec_i = q_rec_i + \
                    (3.0/2.0) * \
                    self.integrate_x(
                        self.data['S_REC']*self.data['ION_TEMPERATURE'])
                q_rec_i = 1e-6 * q_rec_i * self.n_norm * el_charge * self.T_norm / self.t_norm
            except:
                q_rec_i = 0
            if self.neut_temp_eqn:
                q_ion_i = ion_mass/el_mass * \
                    self.integrate_x(
                        self.data['S_ION_SK']*self.data['NEUTRAL_VEL']*self.data['NEUTRAL_VEL'])
                q_ion_i = q_ion_i + \
                    (3.0/2.0) * \
                    self.integrate_x(
                        self.data['S_ION_SK']*self.data['NEUTRAL_TEMPERATURE'])
                q_ion_i = 1e-6 * q_ion_i * \
                    (self.n_norm * el_charge * self.T_norm / self.t_norm)
            else:
                neut_temp = 3.0 / self.T_norm
                q_ion_i = 1e-6 * (self.n_norm * el_charge * self.T_norm / self.t_norm) * (
                    3.0/2.0) * self.integrate_x(self.data['S_ION_SK']*neut_temp)
        else:
            q_ion_i = None
            q_rec_i = None

        # q_n_2dnum
        # q_n_2dnum = 1e-6 * el_charge * self.T_norm * (self.n_norm / self.t_norm) * (ion_mass / el_mass) * (self.integrate_x(self.data['S_ION_SK']*(self.data['NEUTRAL_VEL'] + (self.data['NEUTRAL_VEL_PERP']/np.tan(self.pitch_angle * np.pi / 180.0)))**2) - self.integrate_x(self.data['S_ION_SK']*self.data['NEUTRAL_VEL']**2))
        # print(q_n_2dnum)

        # q_rad
        q_rad = 1e-6 * (self.T_norm * el_charge * self.n_norm *
                        self.v_th / self.x_norm) * (q_rad_deex + q_rad_rec)

        # q_ie
        if (self.ion_temp_eqn):
            try:
                self.load_vars('EI_E_RATE')
                q_ie = 1e-6 * (3.0/2.0) * (el_charge * self.T_norm * self.n_norm * self.v_th /
                                           self.x_norm) * self.integrate_x(self.data['EI_E_RATE']*self.data['DENSITY'])
            except:
                q_ie = 0
        else:
            q_ie = None

        # q_cx
        if (self.ion_temp_eqn and self.neut_temp_eqn):
            try:
                self.load_vars('CX_E_RATE')
                q_cx = -1e-6 * (self.n_norm * el_charge * self.T_norm /
                                self.t_norm) * self.integrate_x(self.data['CX_E_RATE'])
            except:
                q_cx = 0
        else:
            q_cx = None

        return input_power, q_E, q_sh_e, q_sh_i, q_sh_n, q_recyc_n, q_n_e, q_ion_i, q_rec_i, q_rad, q_ie, q_cx

    def q_sh_e(self):
        T_e_boundary = self.data['TEMPERATURE'][-1]
        if self.ion_temp_eqn:
            T_i_boundary = self.data['ION_TEMPERATURE'][-1]
        else:
            T_i_boundary = T_e_boundary
        c_s = sound_speed(T_e_boundary, T_i_boundary)
        n_e_boundary = self.data['DENSITY'][-1] * \
            self.data['DENSITY'][-1] / self.data['DENSITY'][-2]

        if (self.full_fluid and (not self.intrinsic_coupling)):

            gamma_e = 2.0 - 0.5 * \
                np.log(2.0*np.pi*(1.0 + (T_i_boundary/T_e_boundary))
                       * (el_mass/(1.0*ion_mass)))

            # flux_from_T_eqn = (el_charge * self.T_norm * self.v_th) * (((2/3) * self.data['TEMPERATURE'][-1] * c_s) + (
            #     (2/(3*self.data['DENSITY'][-1])) * (gamma_e-2.5) * T_e_boundary * c_s * n_e_boundary))

            # flux_from_n_eqn = (self.v_th * self.n_norm) * c_s * n_e_boundary

            # q_sh_e = (self.n_norm * self.data['DENSITY'][-1]) * ((3/2) + (np.sqrt(el_mass/ion_mass) * (
            #     self.data['FLOW_VEL_X'][-1] / np.sqrt(self.data['TEMPERATURE'][-1])))) * flux_from_T_eqn
            # q_sh_e += (el_charge * self.T_norm * (3/2) *
            #           self.data['TEMPERATURE'][-1] * flux_from_n_eqn)

            # # E advection correction
            # self.load_vars('E_FIELD_X')
            # q_sh_e -= (el_charge * self.T_norm * self.n_norm / self.t_norm) * \
            #     (self.data['E_FIELD_X'][-1] + 0.5*(self.data['E_FIELD_X'][-1] - self.data['E_FIELD_X'][-2])) * self.data['DENSITY'][-1] * \
            #     self.data['FLOW_VEL_X'][-1] * self.dxc[-1]

            # # Pressure gradient correction
            # q_sh_e -= 0.5*el_charge * self.T_norm * self.v_th * self.n_norm * (self.data['DENSITY'][-1] / n_e_boundary) * \
            #     self.data['FLOW_VEL_X'][-1] * self.data['TEMPERATURE'][-1] * \
            #     (self.data['DENSITY'][-1] - n_e_boundary)

            q_sh_e = (el_charge * self.n_norm * self.v_th * self.T_norm) * \
                gamma_e * n_e_boundary * c_s * self.data['TEMPERATURE'][-1]

        else:

            self.load_vars('SHEATH_DATA')
            gamma_e = self.data['SHEATH_DATA'][0]

            q_sh_e = (el_charge * self.n_norm * self.v_th * self.T_norm) * \
                gamma_e * n_e_boundary * c_s * self.data['TEMPERATURE'][-1]

        return q_sh_e

    def q_sh_i(self):
        T_e_boundary = self.data['TEMPERATURE'][-1]
        T_i_boundary = self.data['ION_TEMPERATURE'][-1]
        c_s = sound_speed(T_e_boundary, T_i_boundary)
        n_e_boundary = self.data['DENSITY'][-1] * \
            self.data['DENSITY'][-1] / self.data['DENSITY'][-2]
        gamma_i = 2.5

        q_sh_i = (el_charge * self.n_norm * self.v_th * self.T_norm) * \
            gamma_i * n_e_boundary * c_s * self.data['ION_TEMPERATURE'][-1]
        q_sh_i += (self.v_th ** 3) * self.n_norm * 0.5 * ion_mass * \
            (self.data['ION_VEL'][-1] ** 2) * c_s * n_e_boundary

        # q_sh_i = (el_charge * self.T_norm * self.v_th * self.n_norm) * \
        #     self.data['ION_DENS'][-1] * \
        #     self.data['ION_TEMPERATURE'][-1] * c_s

        # q_sh_i += (el_charge * self.T_norm * self.v_th * self.n_norm) * \
        #     1.5 * self.data['ION_TEMPERATURE'][-1] * c_s * n_e_boundary

        # q_sh_i += (self.v_th ** 3) * self.n_norm * 0.5 * ion_mass * \
        #     (self.data['ION_VEL'][-1] ** 2) * c_s * n_e_boundary

        # q_sh_i += (self.n_norm * self.v_th * el_charge * self.T_norm) * (1/6) * self.data['ION_DENS'][-1] * self.data['ION_VEL'][-1] * (
        #     self.data['ION_TEMPERATURE'][-1] + self.data['TEMPERATURE'][-1] + (self.data['TEMPERATURE'][-1] * (gamma_e - 2.5) * (n_e_boundary / self.data['DENSITY'][-1])))

        # gamma_i = 2.5  # = conductive gamma + 2.5
        # # gamma_i = 2.5*T_i_boundary/T_e_boundary
        # q_sh_i = 1e-6 * (gamma_i) * el_charge * self.T_norm * self.v_th * self.n_norm * \
        #     self.data['ION_TEMPERATURE'][-1] * sound_speed(
        #         self.data['TEMPERATURE'][-1], self.data['ION_TEMPERATURE'][-1]) * self.data['ION_DENS'][-1] * self.data['ION_DENS'][-1] / self.data['ION_DENS'][-2]
        # q_sh_i += 1e-6 * 0.5 * ion_mass * self.n_norm * (self.v_th ** 3) * self.data['ION_VEL'][-1]**2 * sound_speed(
        #     self.data['TEMPERATURE'][-1], self.data['ION_TEMPERATURE'][-1]) * self.data['ION_DENS'][-1] * self.data['ION_DENS'][-1] / self.data['ION_DENS'][-2]

        # # pdV correction
        # q_sh_i += 1e-6 * el_charge * self.T_norm * self.v_th * self.n_norm * \
        #     self.data['ION_TEMPERATURE'][-1] * sound_speed(
        #         self.data['TEMPERATURE'][-1], self.data['ION_TEMPERATURE'][-1]) * self.data['ION_DENS'][-1]

        # # Pressure gradient correction
        # q_sh_i -= 0.5*el_charge * self.T_norm * self.v_th * self.n_norm * (self.data['DENSITY'][-1] / n_e_boundary) * \
        #     self.data['FLOW_VEL_X'][-1] * self.data['ION_TEMPERATURE'][-1] * \
        #     (self.data['DENSITY'][-1] - n_e_boundary)

        # E-field advection correction
        self.load_vars('E_FIELD_X')
        q_sh_i += (el_charge * self.T_norm * self.n_norm / self.t_norm) * \
            2.0*(self.data['E_FIELD_X'][-1] + 0.5*(self.data['E_FIELD_X'][-1] - self.data['E_FIELD_X'][-2])) * self.data['DENSITY'][-1] * \
            self.data['FLOW_VEL_X'][-1] * self.dxc[-1]

        # # KE convection correction
        # q_sh_i -= 1e-6 * 0.5 * ion_mass * self.n_norm * (self.v_th ** 3) * self.data['DENSITY'][-1] * self.data['ION_VEL'][-1] * sound_speed(
        #     self.data['TEMPERATURE'][-1], self.data['ION_TEMPERATURE'][-1]) * (sound_speed(
        #         self.data['TEMPERATURE'][-1], self.data['ION_TEMPERATURE'][-1]) - self.data['ION_VEL'][-2])

        return q_sh_i

    def q_sh_n(self):
        gamma_n = 0.25
        if self.num_neutrals > 1:
            n_n_boundary = np.sum(
                self.data['NEUTRAL_DENS'][-1] * self.data['NEUTRAL_DENS'][-1] / self.data['NEUTRAL_DENS'][-2])
        else:
            n_n_boundary = self.data['NEUTRAL_DENS'][-1] * \
                self.data['NEUTRAL_DENS'][-1] / \
                self.data['NEUTRAL_DENS'][-2]
        q_sh_n = np.sqrt(el_mass/ion_mass) * self.v_th * self.n_norm * el_charge * \
            self.T_norm * gamma_n * n_n_boundary * \
            (self.data['NEUTRAL_TEMPERATURE'][-1] ** 1.5)

        return q_sh_n

    def plot_f(self, harmonic=0, cells=[-1], loglin=True, vs='energy', plotmax=True, lines=None, density_normalise=False):

        if isinstance(cells, int):
            cells = [cells]

        self.load_dist()
        f_norm = self.n_norm / (self.v_th ** 3)
        f = []
        for i in range(len(cells)):
            if density_normalise:
                f.append(self.data['DIST_F'][harmonic][:, cells[i]]
                         * f_norm / self.data['DENSITY'][cells[i]])
            else:
                f.append(self.data['DIST_F'][harmonic][:, cells[i]] * f_norm)
        fig, ax = plt.subplots(1)

        if lines is not None:
            for l in lines:
                ax.axvline(x=l, linestyle='--', color='blue', alpha=0.5)

        if density_normalise:
            label = '$f_' + str(harmonic) + '/n_e$'
        else:
            label = '$f_' + str(harmonic) + '$'
        for i in range(len(cells)):
            if vs == 'energy':
                Egrid = self.T_norm * (self.vgrid/self.v_th) ** 2
                ax.plot(Egrid, f[i], color='red',
                        label=label, alpha=1.0 - 0.5*i)
                ax.set_xlabel('Energy [eV]')
                if plotmax:
                    f0_max = maxwellian(
                        self.data['TEMPERATURE'][cells[i]], self.data['DENSITY'][cells[i]], self.vgrid/self.v_th) * f_norm
                    if density_normalise:
                        f0_max = f0_max / self.data['DENSITY'][cells[i]]
                    ax.plot(Egrid, f0_max, '--', color='black',
                            label='Maxwellian', alpha=1.0 - 0.5*i)
            else:
                ax.plot(self.vgrid / self.v_th,
                        f[i], label=label, alpha=1.0 - 0.5*i)
                ax.set_xlabel('$v / v_{th,0}$')
                if plotmax:
                    f0_max = maxwellian(
                        self.data['TEMPERATURE'][cells[i]], self.data['DENSITY'][cells[i]], self.vgrid/self.v_th) * f_norm
                    if density_normalise:
                        f0_max = f0_max / self.data['DENSITY'][cells[i]]
                    ax.plot(self.vgrid / self.v_th, f0_max, '--',
                            color='black', label='Maxwellian')

        if loglin:
            ax.set_yscale('log')

        ax.set_ylabel(label)
        ax.legend()
        fig.tight_layout(pad=2.0)

    def cell_power_balance(self, species, start_cell=0):

        q = dict()

        if species == 'electrons':
            # q_in (external heating)
            q_in = np.zeros(self.num_x)
            heating_length = (self.xgrid[2*self.n_heating-2] +
                              self.dxc[2*self.n_heating-2]/2.00+self.dxc[0]/2.00)
            for i in range(2*self.n_heating):
                if i % 2 == 0:
                    q_in[i] = 1e6 * self.el_power * \
                        self.dxc[i] / heating_length
            q['in'] = q_in

            # q_L (total heat flow across left cell boundary)
            self.load_vars('HEAT_FLOW_X')
            q_L = np.zeros(self.num_x)
            for i in range(1, self.num_x):
                if i % 2 == 0:
                    q_L[i] = (2.5 * self.n_norm * el_charge * self.T_norm * self.v_th * self.data['DENSITY'][i-1] *
                              self.data['TEMPERATURE'][i-1] *
                              self.data['FLOW_VEL_X'][i-1]) + (el_mass * self.n_norm * (self.v_th ** 3) * self.data['HEAT_FLOW_X'][i-1])
            q['L'] = q_L

            # q_E (volumetric power loss to electric field)
            self.load_vars('E_FIELD_X')
            q_E = np.zeros(self.num_x)
            for i in range(self.num_x):
                if i % 2 == 0:
                    q_E[i] = (2.0 * el_charge * self.T_norm * self.n_norm / self.t_norm) * self.data['E_FIELD_X'][i] * self.data['DENSITY'][i] * \
                        self.data['FLOW_VEL_X'][i] * self.dxc[i]
            q['E'] = q_E

            # q_R (total heat flow across right cell boundary)
            self.load_vars('HEAT_FLOW_X')
            q_R = np.zeros(self.num_x)
            for i in range(self.num_x-1):
                if i % 2 == 0:
                    q_R[i] = (2.5 * self.n_norm * el_charge * self.T_norm * self.v_th * self.data['DENSITY'][i+1] *
                              self.data['TEMPERATURE'][i+1] *
                              self.data['FLOW_VEL_X'][i+1]) + (el_mass * self.n_norm * (self.v_th ** 3) * self.data['HEAT_FLOW_X'][i+1])
            q_R[-1] = self.q_sh_e()
            q['R'] = q_R

            # q_n (volumetric power transfer due to inelastic electron-neutral collisions)
            self.load_vars(['EX_E_RATE', 'DEEX_E_RATE',
                            'ION_E_RATE', 'REC_3B_E_RATE'])
            q_n = np.zeros(self.num_x)
            for i in range(self.num_x):
                if i % 2 == 0:
                    q_n[i] = (self.T_norm * el_charge * self.n_norm * self.v_th / self.x_norm) * (
                        self.data['DEEX_E_RATE'][i] + self.data['REC_3B_E_RATE'][i] - self.data['EX_E_RATE'][i] - self.data['ION_E_RATE'][i]) * self.dxc[i]
            q['n'] = q_n

            # q_ie (volumetric power transfer from ions)
            if (self.ion_temp_eqn):
                self.load_vars('EI_E_RATE')
                q_ie = np.zeros(self.num_x)
                for i in range(self.num_x):
                    if i % 2 == 0:
                        q_ie[i] = (3/2) * (el_charge * self.T_norm * self.n_norm * self.v_th / self.x_norm) * \
                            self.data['EI_E_RATE'][i] * \
                            self.data['DENSITY'][i] * self.dxc[i]
                q['ie'] = q_ie

            # Plot
            fig, ax = plt.subplots(1)
            ax.plot(self.xgrid[start_cell::2], 1e-6 *
                    q_in[start_cell::2], label='$q_{in}$')
            ax.plot(self.xgrid[start_cell::2], 1e-6 *
                    (q_L[start_cell::2] - q_R[start_cell::2]), label='$q_L - q_R$')
            ax.plot(self.xgrid[start_cell::2], 1e-6 *
                    q_n[start_cell::2], label='$q_n$')
            ax.plot(self.xgrid[start_cell::2], -1e-6 *
                    q_E[start_cell::2], label='$q_E$')
            if self.ion_temp_eqn:
                q_tot = (q_L - q_R) + q_n - q_E + q_in + q_ie
                ax.plot(self.xgrid[start_cell::2], 1e-6 *
                        q_ie[start_cell::2], label='$q_{ie}$')
            else:
                q_tot = (q_L - q_R) + q_n - q_E + q_in
            ax.plot(self.xgrid[start_cell::2], 1e-6*q_tot[start_cell::2],
                    color='black', label=r'Discrepancy')
            ax.legend()
            ax.set_xlabel('x [m]')
            ax.set_ylabel('Energy flux [MWm$^{-2}$]')
            q['tot'] = q_tot

        elif species == 'ions':

            # q_in (external heating)
            q_in = np.zeros(self.num_x)
            heating_length = (self.xgrid[2*self.n_heating-2] +
                              self.dxc[2*self.n_heating-2]/2.00+self.dxc[0]/2.00)
            for i in range(2*self.n_heating):
                if i % 2 == 0:
                    q_in[i] = 1e6 * self.ion_power * \
                        self.dxc[i] / heating_length
            q['in'] = q_in

            # q_L (total heat flow across left cell boundary)
            self.load_vars('ION_HEAT_FLOW')
            q_L = np.zeros(self.num_x)
            for i in range(1, self.num_x):
                if i % 2 == 0:
                    # Enthalpy
                    q_L[i] = (2.5 * self.n_norm * el_charge * self.T_norm * self.v_th * self.data['ION_DENS'][i-1] *
                              self.data['ION_TEMPERATURE'][i-1] *
                              self.data['ION_VEL'][i-1])
                    # KE
                    q_L[i] += (0.5 * ion_mass * self.n_norm * (self.v_th ** 3) *
                               self.data['DENSITY'][i-1] * (self.data['ION_VEL'][i-1]**3))
                    # Conductive heat flow
                    q_L[i] += (2.0 * self.n_norm * self.v_th * self.T_norm *
                               el_charge * self.data['ION_HEAT_FLOW'][i-1])
            q['L'] = q_L

            # q_R (total heat flow across right cell boundary)
            self.load_vars('ION_HEAT_FLOW')
            q_R = np.zeros(self.num_x)
            for i in range(self.num_x-1):
                if i % 2 == 0:
                    # Enthalpy
                    q_R[i] = (2.5 * self.n_norm * el_charge * self.T_norm * self.v_th * self.data['ION_DENS'][i+1] *
                              self.data['ION_TEMPERATURE'][i+1] *
                              self.data['ION_VEL'][i+1])
                    # KE
                    q_R[i] += (0.5 * ion_mass * self.n_norm * (self.v_th ** 3) *
                               self.data['DENSITY'][i+1] * (self.data['ION_VEL'][i+1]**3))
                    # Conductive heat flow
                    q_R[i] += (2.0 * self.n_norm * self.v_th * self.T_norm *
                               el_charge * self.data['ION_HEAT_FLOW'][i+1])
            q_R[-1] = self.q_sh_i()

            q['R'] = q_R

            # q_n (volumetric power transfer due to heavy particle exchange)
            self.load_vars(['S_ION_SK', 'S_REC'])
            q_n = np.zeros(self.num_x)
            if self.neut_temp_eqn:
                neut_temp = self.data['NEUTRAL_TEMPERATURE']
                neut_vel = self.data['NEUTRAL_VEL']
            else:
                neut_temp = [3.0 / self.T_norm] * self.num_x
                neut_vel = np.zeros(self.num_x)
            for i in range(self.num_x):
                if i % 2 == 0:
                    q_rec = (ion_mass/el_mass) * self.data['S_REC'][i] * \
                        self.data['ION_VEL'][i] * \
                        self.data['ION_VEL'][i] * self.dxc[i]
                    q_rec = q_rec + \
                        (3.0/2.0) * self.data['S_REC'][i] * \
                        self.data['ION_TEMPERATURE'][i] * self.dxc[i]
                    q_rec *= (self.n_norm * el_charge *
                              self.T_norm / self.t_norm)
                    q_ion = (ion_mass/el_mass) * self.data['S_ION_SK'][i] * \
                        neut_vel[i] * \
                        neut_vel[i] * self.dxc[i]
                    q_ion += (3.0/2.0) * \
                        self.data['S_ION_SK'][i] * neut_temp[i] * self.dxc[i]
                    q_ion *= (self.n_norm * el_charge *
                              self.T_norm / self.t_norm)
                    q_n[i] = q_ion - q_rec
            q['n'] = q_n

            # q_cx
            q_cx = np.zeros(self.num_x)
            if (self.ion_temp_eqn and self.neut_temp_eqn):
                self.load_vars('CX_E_RATE')
                for i in range(self.num_x):
                    if i % 2 == 0:
                        q_cx[i] = (self.n_norm * el_charge * self.T_norm /
                                   self.t_norm) * self.data['CX_E_RATE'][i] * self.dxc[i]
            q['cx'] = q_cx

            # q_E (volumetric power gain from electric field)
            self.load_vars('E_FIELD_X')
            q_E = np.zeros(self.num_x)
            for i in range(self.num_x):
                if i % 2 == 0:
                    q_E[i] = (2.0 * el_charge * self.T_norm * self.n_norm / self.t_norm) * self.data['E_FIELD_X'][i] * self.data['ION_DENS'][i] * \
                        self.data['ION_VEL'][i] * self.dxc[i]
            q['E'] = q_E

            # q_ie (volumetric power transfer to electrons)
            self.load_vars('EI_E_RATE')
            q_ie = np.zeros(self.num_x)
            for i in range(self.num_x):
                if i % 2 == 0:
                    q_ie[i] = (3/2) * (el_charge * self.T_norm * self.n_norm * self.v_th / self.x_norm) * \
                        self.data['EI_E_RATE'][i] * \
                        self.data['DENSITY'][i] * self.dxc[i]
            q['ie'] = q_ie

            # Plot
            fig, ax = plt.subplots(1)
            ax.plot(self.xgrid[start_cell::2], 1e-6 *
                    q_in[start_cell::2], label='$q_{in}$')
            ax.plot(self.xgrid[start_cell::2], 1e-6 *
                    (q_L[start_cell::2] - q_R[start_cell::2]), label='$q_L - q_R$')
            ax.plot(self.xgrid[start_cell::2], 1e-6 *
                    q_n[start_cell::2], label='$q_n$')
            ax.plot(self.xgrid[start_cell::2], 1e-6 *
                    q_E[start_cell::2], label='$q_E$')
            ax.plot(self.xgrid[start_cell::2], -1e-6 *
                    q_ie[start_cell::2], label='$q_{ie}$')
            if self.neut_temp_eqn and self.neut_mom_eqn:
                ax.plot(self.xgrid[start_cell::2], 1e-6 *
                        q_cx[start_cell::2], label='$q_{cx}$')
                q_tot = (q_L - q_R) + q_n + q_E + q_in - q_ie + q_cx
            else:
                q_tot = (q_L - q_R) + q_n + q_E + q_in - q_ie
            ax.plot(self.xgrid[start_cell::2], 1e-6*q_tot[start_cell::2],
                    color='black', label=r'Discrepancy')

            ax.legend()
            ax.set_xlabel('x [m]')
            ax.set_ylabel('Energy flux [MWm$^{-2}$]')
            q['tot'] = q_tot

        return q

    def heat_fluxes(self):

        # q_cond (conductive heat flux)
        self.load_vars('HEAT_FLOW_X')
        q_cond = np.zeros(self.num_x)
        for i in range(self.num_x-1):
            if i % 2 == 0:
                q_cond[i] = el_mass * self.n_norm * \
                    (self.v_th ** 3) * self.data['HEAT_FLOW_X'][i+1]
        if self.full_fluid and (not self.intrinsic_coupling):
            gamma_e = 2.0 - 0.5*np.log(2.0*np.pi*(1.0 + (
                self.data['ION_TEMPERATURE'][-1]/self.data['TEMPERATURE'][-1]))*(el_mass/(1.0*ion_mass)))
        else:
            self.load_vars('SHEATH_DATA')
            gamma_e = self.data['SHEATH_DATA'][0]
        ion_flux, _ = self.get_target_conditions()
        q_cond[-1] = (gamma_e-2.5) * el_charge * self.T_norm * \
            self.data['TEMPERATURE'][-1] * ion_flux

        # q_conv (convective heat flux)
        q_conv = np.zeros(self.num_x)
        for i in range(self.num_x-1):
            if i % 2 == 0:
                q_conv[i] = (2.5 * self.n_norm * el_charge * self.T_norm * self.v_th * self.data['DENSITY'][i+1] *
                             self.data['TEMPERATURE'][i+1] *
                             self.data['FLOW_VEL_X'][i+1])
        q_conv[-1] = 2.5 * el_charge * self.T_norm * \
            self.data['TEMPERATURE'][-1] * ion_flux

        return q_cond, q_conv

    def gamma_e(self):
        if self.full_fluid and (not self.intrinsic_coupling):
            gamma_e = 2.0 - 0.5*np.log(2.0*np.pi*(1.0 + (
                self.data['ION_TEMPERATURE'][-1]/self.data['TEMPERATURE'][-1]))*(el_mass/(1.0*ion_mass)))

        else:
            self.load_vars('SHEATH_DATA')
            gamma_e = self.data['SHEATH_DATA'][0]

        return gamma_e

    def gamma_i(self):

        gamma_i = 2.5
        return gamma_i

    def energy_flux_v_plot(self, intrinsic=True, cell=-1, vs='velocity'):

        self.load_dist()

        f1 = self.data['DIST_F'][1][:, cell]
        f1 *= (self.n_norm / (self.v_th**3))

        q_tot = np.zeros(self.num_v)
        q_v = np.zeros(self.num_v)
        if intrinsic is False:
            # Compute total (conductive + convective) heat flux
            for i, v in enumerate(self.vgrid):
                q_v[i] = el_mass * (2.0 / 3.0) * np.pi * f1[i] * (v**5)

            for i, v in enumerate(self.vgrid):
                q_tot[i] = np.sum(q_v[:i+1] * self.dvc[:i+1])
        else:
            # Compute conductive heat flux
            f0 = self.data['DIST_F'][0][:, cell]
            f0 *= (self.n_norm / (self.v_th**3))
            u_e = self.data['FLOW_VEL_X'][cell] * self.v_th
            n_e = self.data['DENSITY'][cell] * self.n_norm

            if len(self.data['DIST_F']) >= 1:

                f2 = self.data['DIST_F'][2][:, cell]
                f2 *= (self.n_norm / (self.v_th**3))

                for i, v in enumerate(self.vgrid):
                    a = el_mass * (2.0 / 3.0) * np.pi * f1[i] * (v**5)
                    b = el_mass * 2.0 * np.pi * u_e * f0[i] * (v**4)
                    v_u = (4.0 / 3.0) * (np.pi / n_e) * f1[i] * (v**3)
                    U_e = el_mass * 4.0 * np.pi * \
                        f0[i] * (v**4) - (el_mass * n_e * v_u * v_u)
                    P_11 = (
                        (U_e / 3) + ((2/15) * (el_mass * 4.0 * np.pi * f2[i] * (v**4))))
                    c = u_e * P_11
                    q_v[i] = a - b - c

            else:

                for i, v in enumerate(self.vgrid):
                    a = el_mass * (2.0 / 3.0) * np.pi * f1[i] * (v**5)
                    b = el_mass * 2.0 * np.pi * u_e * f0[i] * (v**4)
                    v_u = (4.0 / 3.0) * (np.pi / n_e) * f1[i] * (v**3)
                    U_e = el_mass * 4.0 * np.pi * \
                        f0[i] * (v**4) - (el_mass * n_e * v_u * v_u)
                    P_11 = (U_e / 3)
                    c = u_e * P_11
                    q_v[i] = a - b - c

            for i, v in enumerate(self.vgrid):
                q_tot[i] = np.sum(q_v[:i+1] * self.dvc[:i+1])

        fig, ax = plt.subplots(1)

        if intrinsic:
            integrand_label = r'$\frac{1}{2}m_e w^2 v_{\parallel} f$'
        else:
            integrand_label = r'$\frac{1}{2}m_e v^2 v_{\parallel} f$'

        if vs == 'velocity':

            v_th = np.sqrt(el_charge * self.T_norm *
                           self.data['TEMPERATURE'][cell] / el_mass)
            ax.plot(self.vgrid / v_th, q_v / np.max(q_v), '-', color='red',
                    label=integrand_label)
            ax.plot(self.vgrid / v_th, q_tot / np.max(q_tot),
                    '--', color='red', label='Integral')
            ax.set_xlabel('$v / v_{th}$')

        elif vs == 'energy':

            E = np.zeros(self.num_v)
            for i in range(self.num_v):
                E[i] = self.T_norm * (self.vgrid[i]**2) / (self.v_th ** 2)
            ax.plot(E, q_v / np.max(q_v), '-', color='red',
                    label=integrand_label)
            ax.plot(E, q_tot / np.max(q_tot),
                    '--', color='red', label='Integral')
            ax.set_xlabel('$E [eV]$')

        ax.set_ylabel('a.u.')

        ax.legend()

    def sankey_diagram(self, separate=True):

        input_power, q_E, q_sh_e, q_sh_i, q_sh_n, q_recyc_n, q_n_e, q_ion_i, q_rec_i, q_rad, q_ie, q_cx = self.get_energy_channels()
        # Diffusive neutrals, no T_i equation
        if (self.ion_temp_eqn is False and self.neut_mom_eqn is False):
            # Electrons
            fig, ax = plt.subplots(1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
            sankey = Sankey(ax=ax, scale=0.1, head_angle=150, format='%.2f')
            sankey.add(flows=[input_power, -q_E, -q_sh_e, -q_n_e, -q_rad],
                       labels=['Input', '$q_E$',
                               '$q_{sh,e}$', '$q_{n,e}$', '$q_{rad}$'],
                       orientations=[0, 1, 0, -1, -1],
                       pathlengths=[0.25, 0.25, 0.25, 0.25, 0.25])  # Arguments to matplotlib.patches.PathPatch
            diagrams = sankey.finish()
            ax.set_title('Electron flux channels [MWm$^{-2}$].' + '\n' + r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                self.avg_density/1e19) + r'$\times 10^{19}$m$^{-3}$' + '\nDiscrepancy = {0:.2f} MW/m^2'.format(input_power - (q_sh_e + q_n_e + q_rad + q_E)), size=10)
        # Diffusive neutrals, with T_i equation
        if (self.ion_temp_eqn is True and self.neut_mom_eqn is False):
            if separate is True:
                # Electrons
                # fig = plt.figure()
                # ax = fig.add_subplot(2, 1, 1, xticks=[], yticks=[],title="Electrons [MWm$^{-2}$]")
                fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
                ax[0].set_xticks([])
                ax[0].set_yticks([])
                ax[0].axis('off')
                sankey = Sankey(ax=ax[0], scale=0.1,
                                head_angle=150, format='%.2f')
                sankey.add(flows=[input_power/2, q_ie, -q_E, -q_sh_e, -q_n_e, -q_rad],
                           labels=[
                    'Input', '$q_{ie}$', '$q_E$', '$q_{sh,e}$', '$q_{n,e}$', '$q_{rad}$'],
                    orientations=[0, 1, 1, 0, -1, -1],
                    pathlengths=[0.25, 0.25, 0.25, 0.25, 0.25, 0.25])  # Arguments to matplotlib.patches.PathPatch
                diagrams = sankey.finish()
                ax[0].set_title('Electron flux channels [MWm$^{-2}$].\n' + 'Discrepancy = {0:.2f} MW/m^2'.format(
                    input_power/2 + q_ie - (q_sh_e + q_n_e + q_rad + q_E)))

                # Ions
                ax[1].set_xticks([])
                ax[1].set_yticks([])
                ax[1].axis('off')
                ion_sankey = Sankey(
                    ax=ax[1], scale=0.1, head_angle=150, format='%.2f')
                ion_sankey.add(color='black', flows=[input_power/2, q_E, -q_ie, q_ion_i - q_rec_i, -q_sh_i],
                               labels=[
                    'Input', '$q_{E}$', '$q_{ie}$', '$q_{ion,i}-q_{rec,i}$', '$q_{sh,i}$'],
                    orientations=[0, 1, 1, -1, 0],
                    pathlengths=[0.25, 0.25, 0.25, 0.25, 0.25])  # Arguments to matplotlib.patches.PathPatch
                ion_diagrams = ion_sankey.finish()
                ax[1].set_title('Ion flux channels [MWm$^{-2}$].\n' + 'Discrepancy = {0:.2f}'.format(
                    input_power/2 + q_E + q_ion_i - (q_ie + q_rec_i + q_sh_i)))
            else:
                # Plasma
                fig = plt.figure()
                ax = fig.add_subplot(
                    1, 1, 1, xticks=[], yticks=[], title="Flux channels [MWm$^{-2}$]")
                sankey = Sankey(ax=ax, scale=0.1,
                                head_angle=150, format='%.2f')
                sankey.add(flows=[input_power, q_ion_i, -q_rec_i, -q_sh_i, -q_sh_e, -q_n_e, -q_rad],
                           orientations=[0, -1, -1, 1, 0, -1, -1],
                           labels=['Input', '$q_{ion,i}$', '$q_{rec,i}$',
                                   '$q_{sh,i}$', '$q_{sh,e}$', '$q_{n,e}$', '$q_{rad}$'],
                           pathlengths=[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
                diagrams = sankey.finish()
                ax.set_title('Plasma flux channels [MWm$^{-2}$]. ' + 'Discrepancy = {0:.2f}'.format(
                    input_power + q_ion_i - q_rec_i - q_sh_i - q_sh_e - q_n_e - q_rad))

        # Fluid neutrals, with T_n equation
        if (self.ion_temp_eqn is True and self.neut_mom_eqn is True and self.neut_temp_eqn is True):
            if self.periodic is True:
                # Electrons
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1, xticks=[],
                                     yticks=[], title="Electrons [MWm$^{-2}$]")
                sankey = Sankey(ax=ax, scale=0.1,
                                head_angle=150, format='%.2f')
                sankey.add(flows=[input_power/2, q_ie, -q_n_e, -q_rad],
                           labels=['Input', '$q_{ie}$',
                                   '$q_{n,e}$', '$q_{rad}$'],
                           orientations=[0, 1, -1, -1],
                           pathlengths=[0.25, 0.25, 0.25, 0.25])  # Arguments to matplotlib.patches.PathPatch
                diagrams = sankey.finish()
                ax.set_title('Electron flux channels [MWm$^{-2}$]. ' + 'Discrepancy = {0:.2f}'.format(
                    input_power/2 + q_ie - (q_n_e + q_rad)))

                # Ions
                ion_fig = plt.figure()
                ion_ax = ion_fig.add_subplot(
                    1, 1, 1, xticks=[], yticks=[], title="Ions [MWm$^{-2}$]")
                ion_sankey = Sankey(ax=ion_ax, scale=0.1,
                                    head_angle=150, format='%.2f')
                ion_sankey.add(color='black', flows=[input_power/2, q_ion_i, -q_ie, -q_rec_i, -q_cx],
                               labels=[
                    'Input', '$q_{ion,i}$', '$q_{ie}$', '$q_{rec,i}$', '$q_{cx}$'],
                    orientations=[0, -1, 1, -1, -1],
                    pathlengths=[0.25, 0.25, 0.25, 0.25, 0.25])  # Arguments to matplotlib.patches.PathPatch
                ion_diagrams = ion_sankey.finish()
                ion_ax.set_title('Ion flux channels [MWm$^{-2}$]. ' + 'Discrepancy = {0:.2f}'.format(
                    input_power/2 + q_ion_i - (q_ie + q_rec_i + q_cx)))

                # Neutrals
                neut_fig = plt.figure()
                neut_ax = neut_fig.add_subplot(
                    1, 1, 1, xticks=[], yticks=[], title="Ions [MWm$^{-2}$]")
                neut_sankey = Sankey(ax=neut_ax, scale=0.1,
                                     offset=0.2, head_angle=150, format='%.2f')
                neut_sankey.add(color='black', flows=[q_cx, q_rec_i, -q_ion_i],
                                labels=['$q_{cx}$',
                                        '$q_{rec,i}$', '$q_{ion,i}$'],
                                orientations=[0, 1, 1])  # Arguments to matplotlib.patches.PathPatch
                neut_diagrams = neut_sankey.finish()
                neut_ax.set_title(
                    'Neutral flux channels [MWm$^{-2}$]. ' + 'Discrepancy = {0:.2f}'.format(q_cx + q_rec_i - (q_ion_i)))
                return
            else:
                pass

            if separate is True:

                fig, ax = plt.subplots(2, 2, figsize=(9.5, 8))
                ax[0, 0].set_xticks([])
                ax[0, 0].set_yticks([])
                ax[0, 0].axis('off')
                ax[1, 0].set_xticks([])
                ax[1, 0].set_yticks([])
                ax[1, 0].axis('off')
                ax[0, 1].set_xticks([])
                ax[0, 1].set_yticks([])
                ax[0, 1].axis('off')
                ax[1, 1].set_xticks([])
                ax[1, 1].set_yticks([])
                ax[1, 1].axis('off')

                # Electrons
                # fig = plt.figure()
                # ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],title="Electrons [MWm$^{-2}$]")
                sankey = Sankey(ax=ax[0, 0], scale=0.1,
                                offset=0.2, head_angle=150, format='%.2f')
                sankey.add(flows=[input_power/2, q_ie, -q_sh_e, -q_n_e, -q_rad, -q_E],
                           labels=['input', 'i-e transfer', 'sheath',
                                   'given to neutrals', 'radiation', 'E-field'],
                           orientations=[0, 1, 0, -1, -1, 1],
                           pathlengths=[0.25, 0.25, 0.25, 0.75, 0.25, 0.25])  # Arguments to matplotlib.patches.PathPatch
                diagrams = sankey.finish()
                ax[0, 0].set_title('Electron flux channels [MWm$^{-2}$].\n' + 'Discrepancy = {0:.2f}'.format(
                    input_power/2 + q_ie - (q_sh_e + q_n_e + q_rad + q_E)))

                # Ions
                ion_sankey = Sankey(
                    ax=ax[0, 1], scale=0.1, offset=0.2, head_angle=150, format='%.2f')
                ion_sankey.add(color='black', flows=[input_power/2, q_E, -q_ie, -q_sh_i, -q_cx + q_ion_i - q_rec_i],
                               labels=['input', 'E-field',
                                       'i-e transfer', 'sheath', 'CX, etc'],
                               orientations=[0, 1, 1, 0, -1],
                               pathlengths=[0.25, 0.25, 0.25, 0.25, 0.25])  # Arguments to matplotlib.patches.PathPatch
                ion_diagrams = ion_sankey.finish()
                ax[0, 1].set_title('Ion flux channels [MWm$^{-2}$].\n' + 'Discrepancy = {0:.2f}'.format(
                    input_power/2 + q_E + q_ion_i - (q_ie + q_rec_i + q_sh_i + q_cx)))

                # Neutrals
                neut_sankey = Sankey(
                    ax=ax[1, 0], scale=0.1, offset=0.2, head_angle=150, format='%.2f')
                neut_sankey.add(color='black', flows=[q_recyc_n, q_cx + q_rec_i - q_ion_i, -q_sh_n],
                                labels=['recycling flux', 'CX, etc', 'sheath'],
                                orientations=[-1, 0, 0],
                                pathlengths=[0.25, 0.25, 0.25])  # Arguments to matplotlib.patches.PathPatch
                neut_diagrams = neut_sankey.finish()
                ax[1, 0].set_title('Neutral flux channels [MWm$^{-2}$].\n' + 'Discrepancy = {0:.2f}'.format(
                    q_cx + q_rec_i + q_recyc_n - (q_ion_i + q_sh_n)))

                fig.suptitle(r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                    self.avg_density/1e19) + r'$\times 10^{19}$m$^{-3}$')
                return
            else:
                # Plasma
                fig = plt.figure()
                ax = fig.add_subplot(
                    1, 1, 1, xticks=[], yticks=[], title="Flux channels [MWm$^{-2}$]")
                sankey = Sankey(ax=ax, scale=0.1,
                                head_angle=150, format='%.2f')
                sankey.add(flows=[input_power, q_ion_i-q_rec_i, -q_cx, -q_sh_i, -q_sh_e, -q_n_e, -q_rad],
                           orientations=[0, -1, -1, 1, 0, -1, -1],
                           labels=['Input', '$q_{ion,i} - q_{rec,i}$', '$q_{cx}$',
                                   '$q_{sh,i}$', '$q_{sh,e}$', '$q_{n,e}$', '$q_{rad}$'],
                           pathlengths=[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
                diagrams = sankey.finish()
                ax.set_title('Flux channels in [MWm$^{-2}$]. ' + 'Discrepancy = {0:.2f}'.format(
                    input_power + q_ion_i - q_rec_i - q_sh_i - q_sh_e - q_n_e - q_rad - q_cx))
                return

    def get_input_power(self):
        # Return input power (normalised to MW/m^2)
        el_power = 0.0
        ion_power = 0.0
        with open(os.path.join(self.dir, 'INPUT', 'NEUT_AND_HEAT_INPUT.txt')) as f:
            l = f.readlines()
            for line in l:
                if 'HEAT_POWER' in line and 'ION' not in line:
                    el_power = float(line.split()[2])
                if 'ION_HEAT_POWER' in line:
                    ion_power = float(line.split()[2])
                if 'N_HEATING' in line:
                    if line.split()[0] == 'N_HEATING':
                        self.n_heating = int(line.split()[2])
        self.el_power = el_power
        self.ion_power = ion_power

        return el_power, ion_power

    def reconstruct_f_plot(self, cart_transform=False, cell=-1, num_contours=10, loglin=True, max_vth=10, vs='energy'):

        # Reconstruct f in polar coordinates
        angular_samples = self.num_v * 2
        theta_grid = np.linspace(0, 2*np.pi, angular_samples)
        E_grid = np.zeros(self.num_v)
        for i in range(self.num_v):
            E_grid[i] = self.T_norm * (self.vgrid[i] / self.v_th) ** 2
        self.load_dist()
        f_pol = np.zeros([self.num_v, angular_samples])
        for l in range(self.l_max+1):
            for i, theta in enumerate(theta_grid):
                f_pol[:, i] += self.data['DIST_F'][l][:, cell] * \
                    legendre_coeff(l, np.cos(theta))

        if cart_transform:
            # Transform to cartesian coordinates
            T_e = self.data['TEMPERATURE'][cell] * self.T_norm
            v_th = np.sqrt(2.0 * T_e * el_charge / el_mass)
            cart_num_v = self.num_v

            vgrid_par = np.linspace(-self.vgrid[-1],
                                    self.vgrid[-1], 2*cart_num_v) / v_th
            vgrid_perp = np.linspace(0.0, self.vgrid[-1], cart_num_v) / v_th

            vgrid_norm = self.vgrid / v_th
            f_cart = np.zeros([len(vgrid_par), len(vgrid_perp)])

            for i in range(len(vgrid_par)):
                for j in range(len(vgrid_perp)):
                    v_par = vgrid_par[i]
                    v_perp = vgrid_perp[j]
                    v = np.sqrt(v_par**2 + v_perp**2)
                    th = np.arctan2(v_perp, v_par)

                    f_cart[i, j] = interp_f(
                        f_pol, v, th, vgrid_norm, theta_grid)

            f_cart[cart_num_v:, :] += f_cart[:cart_num_v, :]

            # Plot
            fig, ax = plt.subplots(1)

            minvpar_idx = bisect.bisect(vgrid_par, -max_vth)
            maxvpar_idx = bisect.bisect(vgrid_par, max_vth)
            maxvperp_idx = bisect.bisect(vgrid_perp, max_vth)

            indep_gridx = vgrid_par[minvpar_idx:maxvpar_idx]
            indep_gridy = vgrid_perp[:maxvperp_idx]
            ax.set_xlabel('$v_{\parallel}/v_{th}$')
            ax.set_ylabel('$v_{\perp}/v_{th}$')

            if loglin:
                f_cart_plot = np.transpose(
                    np.log(f_cart[minvpar_idx:maxvpar_idx, :maxvperp_idx]))
            else:
                f_cart_plot = np.transpose(
                    f_cart[minvpar_idx:maxvpar_idx, :maxvperp_idx])

            cs = ax.contourf(indep_gridx, indep_gridy,
                             f_cart_plot, levels=num_contours)

            cbar = fig.colorbar(cs)
            plt.show()

        else:

            # Plot
            T_e = self.data['TEMPERATURE'][cell] * self.T_norm
            v_th = np.sqrt(2.0 * T_e * el_charge / el_mass)
            vgrid_norm = self.vgrid / v_th
            Egrid_norm = vgrid_norm**2
            maxv_idx = bisect.bisect(vgrid_norm, max_vth)

            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
            ax.set_title(r'$f(v,\theta)$')
            if vs == 'energy':
                indep_grid = Egrid_norm[:maxv_idx]
                yticks = np.arange(1, max_vth, 2) * self.T_norm
                ax.set_yticks(yticks)
                yticklabels = [str(int(tick)) for tick in yticks]
                yticklabels[0] = yticklabels[0] + '$T_e$'
                ax.set_yticklabels(yticklabels)
            elif vs == 'velocity':
                indep_grid = vgrid_norm[:maxv_idx]
                yticks = np.arange(2, max_vth, 2)
                ax.set_yticks(yticks)
                yticklabels = [str(int(tick)) for tick in yticks]
                yticklabels[0] = yticklabels[0] + '$v_{th}$'
                ax.set_yticklabels(yticklabels)

            f_pol_plot = f_pol[:maxv_idx, :]
            if loglin:
                min_f = f_pol_plot.min()
                if min_f <= 0.0:
                    min_f = 1e-30
                lev_exp = np.arange(np.floor(np.log10(min_f)-1),
                                    np.ceil(np.log10(f_pol_plot.max())+1))
                levs = np.power(10, lev_exp)
                cs = ax.contourf(theta_grid, indep_grid,
                                 f_pol_plot, levs, norm=colors.LogNorm())
            else:
                cs = ax.contourf(theta_grid, indep_grid,
                                 f_pol_plot, levels=num_contours)

            cbar = fig.colorbar(cs, pad=0.1)
            plt.show()
            ax.set_xticklabels(
                ['$0\degree$', '', '$90\degree$', '', '$180\degree$', '', '$270\degree$'])

    def get_collisionality(self):

        # Calculate nu star
        upstream_mfp = 1e16 * \
            ((self.data['TEMPERATURE'][0] * self.T_norm) ** 2) / \
            (self.data['DENSITY'][0] * self.n_norm)
        nu_star = self.connection_length / upstream_mfp
        return nu_star

    def get_target_collisionality(self):

        # Calculate nu star
        t_mfp = 1e16 * \
            ((self.data['TEMPERATURE'][-1] * self.T_norm) ** 2) / \
            (self.data['DENSITY'][-1] * self.n_norm)
        nu_t = 1 / t_mfp
        return nu_t

    def ion_rate(self):

        self.load_vars(['S_ION_SK'])

        S_ion = self.n_norm * (self.v_th / self.x_norm) * self.data['S_ION_SK']
        try:
            n_n = np.sum(self.data['NEUTRAL_DENS'], 1) * self.n_norm
            # n_n = self.data['NEUTRAL_DENS'][:,0] * self.n_norm
        except(np.AxisError):
            n_n = self.data['NEUTRAL_DENS'] * self.n_norm

        n_e = self.data['DENSITY'] * self.n_norm

        T_e = self.data['TEMPERATURE'] * self.T_norm

        sigma_v = np.zeros(self.num_x)
        for i in range(self.num_x):
            if (n_n[i] > 0.0):
                sigma_v[i] = S_ion[i] / (n_n[i] * n_e[i])
        fig, ax = plt.subplots(1)

        compare_sd1d = False
        if compare_sd1d:
            sd1d_ion_rate = SD1D_ion_rate()
            ax.plot(sd1d_ion_rate[:, 0],
                    sd1d_ion_rate[:, 1], '--', label='SD1D')

        compare_adas = False
        if compare_adas:
            adas_ion_rate = ADAS_ion_rate()
            ax.plot(adas_ion_rate[:10, 0],
                    adas_ion_rate[:10, 1], '.-', label='ADAS')

        compare_amjuel = True
        if compare_amjuel:
            amjuel_ion_rate = np.zeros(self.num_x)
            for i in range(self.num_x):
                amjuel_ion_rate[i] = amjuel_rate('ionisation', n_e[i], T_e[i])
            ax.plot(T_e[2:-2], amjuel_ion_rate[2:-2], '--', label='AMJUEL')

        ax.plot(T_e[2:-2], sigma_v[2:-2], label='SOL-KiT')
        ax.set_xlabel('$T_e$ [eV]')
        ax.set_ylabel(r'$\langle \sigma v \rangle$ [m$^3$ s$^{-1}$]')
        ax.set_title(r'$n_{e} $' + ' = {:1.0f}'.format(
            n_e[1]/1e19) + r'$\times 10^{19}$m$^{-3}$', loc='left')
        ax.legend()
        ax.grid()
        # plt.xscale('log')
        plt.yscale('log')

    def ion_ex_E_rate(self, compare_sd1d=True):

        self.load_vars(['EX_E_RATE', 'DEEX_E_RATE',
                        'ION_E_RATE', 'REC_3B_E_RATE'])
        q_n = (self.T_norm * self.n_norm * self.v_th / self.x_norm) * \
            (self.data['EX_E_RATE'] + self.data['ION_E_RATE'] -
             self.data['DEEX_E_RATE'] - self.data['REC_3B_E_RATE'])
        q_n_exonly = (self.T_norm * self.n_norm * self.v_th / self.x_norm) * \
            (self.data['EX_E_RATE'] - self.data['DEEX_E_RATE'])
        try:
            n_n = np.sum(self.data['NEUTRAL_DENS'], 1) * self.n_norm
        except:
            n_n = self.data['NEUTRAL_DENS'] * self.n_norm
        n_e = self.data['DENSITY'] * self.n_norm
        T_e = self.data['TEMPERATURE'] * self.T_norm
        sigma_v_eps = q_n / (n_n * n_e)
        sigma_v_eps_exonly = q_n_exonly / (n_n * n_e)

        if compare_sd1d:
            Te = np.logspace(0, 1.5, num=50)
            TT = Te
            Y = 10.2 / TT
            fEXC = (49.0E-14/(0.28+Y))*np.exp(-Y)*np.sqrt(Y*(1.0+Y))

        # Plot
        fig, ax = plt.subplots(1)
        if (compare_sd1d):
            ax.plot(Te, fEXC, label='SD1D')
        ax.plot(T_e[::2], sigma_v_eps[::2], label='SOL-KiT')
        ax.plot(T_e[::2], sigma_v_eps_exonly[::2],
                '--', label='SOL-KiT (ex+deex only)')
        ax.set_xlabel('$T_e$ [eV]')
        ax.set_ylabel(
            r'$\langle \sigma v \varepsilon \rangle$ [eV m$^3$ s$^{-1}$]')
        ax.legend()
        ax.grid()
        plt.yscale('log')

    def ionisation_degree_plot(self, compare_amjuel=True):
        try:
            n_n = np.sum(self.data['NEUTRAL_DENS'], 1) * self.n_norm
        except:
            n_n = self.data['NEUTRAL_DENS'] * self.n_norm
        n_e = self.data['DENSITY'] * self.n_norm
        T_e = self.data['TEMPERATURE'] * self.T_norm

        sk_iz_degree = 100.0*n_e / (n_e + n_n)

        amjuel_iz_degree = np.zeros(len(T_e))
        for i in range(len(T_e)):
            amjuel_iz_degree[i] = 100.0 / (1 + (amjuel_rate(
                'recombination', 0, n_e[i], T_e[i]) / amjuel_rate('ionisation', 0, n_e[i], T_e[i])))

        sd1d_iz_degree = np.zeros(len(T_e))
        for i in range(len(T_e)):
            sd1d_iz_degree[i] = 100.0 / (1 + (amjuel_rate(
                'recombination', 0, 1e14, T_e[i]) / amjuel_rate('ionisation', 0, 1e14, T_e[i])))

        # Plot
        fig, ax = plt.subplots(1)
        ax.plot(T_e[2:-2], sk_iz_degree[2:-2], label='SOL-KiT')
        ax.plot(T_e[2:-2], sd1d_iz_degree[2:-2], '--', label='SD1D')
        ax.plot(T_e[2:-2], amjuel_iz_degree[2:-2], '--', label='AMJUEL')
        ax.set_xlabel('$T_e$ [eV]')
        ax.set_ylabel(
            r'Ionisation degree (%)')
        ax.legend()
        ax.grid()
        # plt.yscale('log')
        # plt.xscale('log')

    def neutral_distribution(self, cells=None, degeneracy_norm=False, energy_axis=False):

        fig, ax = plt.subplots(1)
        plt.yscale('log')

        if energy_axis:
            x = [-13.6 / (b+1)**2 for b in range(self.num_neutrals)]
            ax.set_xlabel('Level energy [eV]')
        else:
            x = np.arange(self.num_neutrals)
            ax.set_xlabel('Level (principal quantum number)')

        ax.set_ylabel('Density  / $n_0$')

        # TODO: compare with boltzmann neuts

        if cells is None:
            cells = [-1]
        elif isinstance(cells, int):
            cells = [cells]

        for cell in cells:
            n_n = self.data['NEUTRAL_DENS'][cell]
            if degeneracy_norm:
                for b in range(len(n_n)):
                    n_n[b] = n_n[b] / (2.0*(b+1)**2)
            label = '{:1.1f}m'.format(self.xgrid[cell] + self.xgrid[1])
            ax.plot(x, n_n, label=label)

        ax.legend()

    def q_n_comparison_plot(self, compare_sd1d=False, compare_amjuel=True):
        fig, ax = plt.subplots(1)

        self.load_vars(['EX_E_RATE', 'DEEX_E_RATE',
                        'ION_E_RATE', 'REC_3B_E_RATE'])
        q_n_sk = 1e-6*(self.T_norm * el_charge * self.n_norm * self.v_th / self.x_norm) * \
            (self.data['EX_E_RATE'] + self.data['ION_E_RATE'] -
             self.data['DEEX_E_RATE'] - self.data['REC_3B_E_RATE'])

        try:
            n_n = np.sum(self.data['NEUTRAL_DENS'], 1) * self.n_norm
        except:
            n_n = self.data['NEUTRAL_DENS'] * self.n_norm
        n_e = self.data['DENSITY'] * self.n_norm
        T_e = self.data['TEMPERATURE'] * self.T_norm

        if compare_sd1d:
            q_n_sd1d = np.zeros(self.num_x)
            for i in range(self.num_x):
                TT = T_e[i]
                Y = 10.2 / TT
                q_n_sd1d[i] = 1e-6*el_charge * (49.0E-14/(0.28+Y))*np.exp(-Y) * \
                    np.sqrt(Y*(1.0+Y)) * n_e[i] * n_n[i]
                # Need to account for recombination here

        if compare_amjuel:
            q_n_amjuel = np.zeros(self.num_x)
            for i in range(self.num_x):
                q_n_amjuel_ionex = amjuel_rate(
                    'ionisation', 2, n_e[i], T_e[i]) * n_e[i] * n_n[i]
                q_n_amjuel_rec = amjuel_rate(
                    'recombination', 2, n_e[i], T_e[i]) * n_e[i] * n_e[i]
                q_n_amjuel_rec -= 13.6 * \
                    amjuel_rate('recombination', 0,
                                n_e[i], T_e[i]) * n_e[i] * n_e[i]

                q_n_amjuel[i] = 1e-6 * el_charge * \
                    (q_n_amjuel_ionex - q_n_amjuel_rec)

        ax.plot(self.xgrid, q_n_sk, label='SOL-KiT')
        if compare_sd1d:
            ax.plot(self.xgrid, q_n_sd1d, '--', label='SD1D')
        if compare_amjuel:
            ax.plot(self.xgrid, q_n_amjuel, '--', label='AMJUEL')
        ax.legend()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('$q_n$ [MWm$^{-3}$]')

    def part_source_comparison_plot(self, compare_sd1d=True, compare_amjuel=True):
        fig, ax = plt.subplots(1)

        self.load_vars(['S_ION_SK', 'S_REC'])
        s_sk = (self.data['S_ION_SK'] - self.data['S_REC']) * \
            self.n_norm / self.t_norm

        try:
            n_n = np.sum(self.data['NEUTRAL_DENS'], 1) * self.n_norm
        except:
            n_n = self.data['NEUTRAL_DENS'] * self.n_norm
        n_e = self.data['DENSITY'] * self.n_norm
        T_e = self.data['TEMPERATURE'] * self.T_norm

        if compare_sd1d:
            s_sd1d = np.zeros(self.num_x)
            for i in range(self.num_x):
                s_sd1d[i] = amjuel_rate(
                    'ionisation', 0, 1e14, T_e[i]) * n_e[i] * n_n[i]
                s_sd1d[i] -= amjuel_rate('recombination',
                                         0, 1e14, T_e[i]) * n_e[i] * n_e[i]

        if compare_amjuel:
            s_amjuel = np.zeros(self.num_x)
            for i in range(self.num_x):
                s_amjuel[i] = amjuel_rate(
                    'ionisation', 0, n_e[i], T_e[i]) * n_e[i] * n_n[i]
                s_amjuel[i] -= amjuel_rate('recombination',
                                           0, n_e[i], T_e[i]) * n_e[i] * n_e[i]

        ax.plot(self.xgrid[::2], s_sk[::2], label='SOL-KiT')
        if compare_sd1d:
            ax.plot(self.xgrid[::2], s_sd1d[::2], '--', label='SD1D')
        if compare_amjuel:
            ax.plot(self.xgrid[::2], s_amjuel[::2], '--', label='AMJUEL')
        ax.legend()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('$S_{ion} - S_{rec}$ [m$^{-3}s^{-1}$]')


class SKTRun(SKRun):
    def __init__(self, run_dir):
        SKRun.__init__(self, run_dir)

        self.top_dir = os.path.dirname(self.dir)
        self.get_runs()
        self.get_timestamps()
        self.tdata = {}

    def get_timestamps(self):
        # Get NUM_TIMESTEPS from each run
        self.num_timesteps = []
        self.dts = []
        for run_num in self.run_nums:
            gridfile = os.path.join(
                self.top_dir, run_num, 'INPUT', 'GRID_INPUT.txt')
            with open(gridfile) as f:
                lines = f.readlines()
                for l in lines:
                    if len(l.split()) > 0:
                        if l.split()[0] == 'TIMESTEP_NUM':
                            self.num_timesteps.append(int(l.split()[2]))
                        if l.split()[0] == 'dt':
                            self.dts.append(float(l.split()[2]))

        # Get the timestamps of saved data
        timestamps = []
        for i, run_num in enumerate(self.run_nums):
            saved_files = os.listdir(os.path.join(
                self.top_dir, run_num, 'OUTPUT', 'DENSITY'))
            saved_timesteps = sorted([int(f[8:-4])
                                     for f in saved_files if 'DENSITY_' in f])
            if i == 0:
                timestamps = timestamps + \
                    [t * self.dts[i] * self.t_norm for t in saved_timesteps]
            else:
                prev_ts = timestamps[-1]
                timestamps = timestamps + \
                    [(t * self.dts[i] * self.t_norm) +
                     prev_ts for t in saved_timesteps]

        self.timestamps = np.array(timestamps)
        self.tot_timesteps = len(self.timestamps)

    def get_runs(self):
        runs_list = os.listdir(self.top_dir)
        self.run_nums = sorted(
            [d for d in runs_list if 'Run_' in d], key=lambda d: int(d.strip('Run_')))
        self.num_runs = len(self.run_nums)

    # def load_tdist(self):
    #     var_dir = os.path.join(self.dir, 'OUTPUT', 'DIST_F')
    #     var_files = sorted(os.listdir(var_dir))
    #     self.data['DIST_F'] = [None] * (self.l_max + 1)
    #     for l in range(self.l_max+1):
    #         # print(var_files[-11:-9], str(l) + '_')
    #         l_files = sorted([
    #             f for f in var_files if f[-11:-9] == str(l) + '_'])
    #         self.data['DIST_F'][l] = np.loadtxt(
    #             os.path.join(var_dir, l_files[-1]))

    def load_tvars(self, variables, normalised=False):
        if type(variables) == str:
            variables = [variables]

        for variable in variables:
            if variable not in self.tdata.keys():

                if variable == 'DIST_F':
                    try:
                        dist_files = []
                        self.tdata['DIST_F'] = np.zeros(
                            [len(self.timestamps), self.l_max+1, self.num_v, self.num_x])

                        for run_num in self.run_nums:
                            var_dir = os.path.join(
                                self.top_dir, run_num, 'OUTPUT', variable)
                            dist_files = dist_files + [os.path.join(var_dir, vf) for vf in os.listdir(
                                var_dir) if os.path.basename(vf).startswith('F_L  ')]

                        for l in range(self.l_max+1):
                            l_files = sorted([f for f in dist_files if os.path.basename(f).startswith(
                                'F_L  ' + str(l) + '_')], key=lambda x: (float((x.split(os.sep))[-4].strip('Run_')), int(os.path.basename(x)[-9:-4])))
                            for i, l_file in enumerate(l_files):
                                self.tdata['DIST_F'][i, l, :,
                                                     :] = np.loadtxt(l_file)

                    except(FileNotFoundError):
                        print('WARNING: Variable ' +
                              variable + ' not found in output')
                else:
                    # Read in raw data
                    try:
                        var_files = []
                        for run_num in self.run_nums:
                            var_dir = os.path.join(
                                self.top_dir, run_num, 'OUTPUT', variable)
                            var_files = var_files + sorted(
                                [os.path.join(var_dir, vf) for vf in os.listdir(var_dir) if variable in vf])

                        if variable == 'SHEATH_DATA':
                            self.tdata[variable] = np.zeros(
                                [len(self.timestamps), 5])
                        else:
                            self.tdata[variable] = np.zeros(
                                [len(self.timestamps), self.num_x])

                        for i, vf in enumerate(var_files):
                            if variable == 'NEUTRAL_DENS':
                                self.tdata[variable][i, :] = np.sum(
                                    np.loadtxt(vf), 1)
                            else:
                                self.tdata[variable][i, :] = np.loadtxt(vf)

                    except(FileNotFoundError):
                        print('WARNING: Variable ' +
                              variable + ' not found in output')

                # De-normalise
                if normalised is False:
                    self.tdata[variable] *= self.get_var_norm(variable)

    def animate(self, variable, interval=100):

        self.load_tvars(variable)

        fig, ax = plt.subplots(1)
        line, = ax.plot(self.xgrid, self.tdata[variable][0, :])
        ax.set_ylim([np.min(self.tdata[variable]),
                    np.max(self.tdata[variable])])
        ax.set_xlabel('x [m]')
        ax.set_ylabel(self.get_var_label(variable))

        def update_line(i):
            line.set_data(self.xgrid, self.tdata[variable][i, :])
            ax.set_title('{:.2f} $\mu$s'.format(1e6 * self.timestamps[i]))

            return line,

        anim = animation.FuncAnimation(
            fig, update_line, frames=self.tot_timesteps, interval=interval, blit=False, repeat=True)
        plt.show()

        return anim

    def get_target_density(self, i):
        self.load_tvars('DENSITY')
        return self.tdata['DENSITY'][i, -1] * self.tdata['DENSITY'][i, -1] / self.tdata['DENSITY'][i, -2]

    def get_target_flux(self, i):
        self.load_tvars('TEMPERATURE', 'ION_TEMPERATURE')
        T_e_boundary = self.tdata['TEMPERATURE'][i, -1] / self.T_norm
        T_i_boundary = self.tdata['ION_TEMPERATURE'][i, -1] / self.T_norm
        n_e_boundary = self.get_target_density(i)
        flux = n_e_boundary * \
            sound_speed(T_e_boundary, T_i_boundary) * self.v_th
        return flux

    def q_sh_e(self, i):

        T_e_boundary = self.tdata['TEMPERATURE'][i, -1]
        if self.ion_temp_eqn:
            T_i_boundary = self.tdata['ION_TEMPERATURE'][i, -1]
        else:
            T_i_boundary = T_e_boundary
        c_s = sound_speed(T_e_boundary/self.T_norm,
                          T_i_boundary/self.T_norm) * self.v_th
        n_e_boundary = self.tdata['DENSITY'][i, -1] * \
            self.tdata['DENSITY'][i, -1] / self.tdata['DENSITY'][i, -2]

        if (self.full_fluid and (not self.intrinsic_coupling)):

            gamma_e = 2.0 - 0.5 * \
                np.log(2.0*np.pi*(1.0 + (T_i_boundary/T_e_boundary))
                       * (el_mass/(1.0*ion_mass)))

        else:

            self.load_tvars('SHEATH_DATA')
            gamma_e = self.tdata['SHEATH_DATA'][i, 0]

        q_sh_e = el_charge * gamma_e * n_e_boundary * \
            c_s * self.tdata['TEMPERATURE'][i, -1]

        return q_sh_e

    def timeseries(self, variable, cell=-1):

        self.load_tvars(variable)
        var_tseries = self.tdata[variable][:, cell]

        fig, ax = plt.subplots(1)
        ax.plot(self.timestamps*1e6, var_tseries)
        ax.set_xlabel('Time [$\mu$s]')
        ax.set_ylabel(self.get_var_label(variable))
        ax.set_title(variable + ' at x={:.2f}m'.format(self.xgrid[cell]))

    def burnthrough_plot(self):

        self.load_tvars(['DENSITY', 'NEUTRAL_DENS'])
        ionization_degree = np.zeros(self.tot_timesteps)
        for i, t in enumerate(self.timestamps):
            avg_dens_plasma = np.sum(
                self.tdata['DENSITY'][i, :] * self.dxc) / self.connection_length
            avg_dens_neuts = np.sum(
                self.tdata['NEUTRAL_DENS'][i, :] * self.dxc) / self.connection_length
            ionization_degree[i] = 100.0 * \
                (1.0 - avg_dens_neuts / avg_dens_plasma)

        fig, ax = plt.subplots(1)
        ax.plot(self.timestamps*1e6, ionization_degree)
        ax.set_ylabel('Ionization degree [%]')
        ax.set_xlabel('Time [$\mu$s]')
        ax.set_title('Line-averaged plasma ionization degree')


class SKTRundeck(SKRundeck):
    def __init__(self, rundeck_dir, identifier='Output_'):
        SKRundeck.__init__(self, rundeck_dir, identifier=identifier)
        self.load_trundeck(identifier)

    def load_trundeck(self, identifier):
        self.truns = []
        for i, r in enumerate(self.runs):
            self.truns.append(SKTRun(os.path.dirname(r.dir)))

    def animate(self, variable, interval=5):

        for trun in self.truns:
            trun.load_tvars(variable)

        # Set up figure and colour map
        fig, ax = plt.subplots(1)

        # Get timestamp and variable limits
        min_ts = np.min(self.truns[0].timestamps)
        max_ts = np.max(self.truns[0].timestamps)
        max_num_ts = self.truns[0].tot_timesteps
        min_val = np.min(self.truns[0].tdata[variable])
        max_val = np.max(self.truns[0].tdata[variable])
        for trun in self.truns:
            if np.min(trun.timestamps) < min_ts:
                min_ts = np.min(trun.timestamps)
            if np.max(trun.timestamps) > max_ts:
                max_ts = np.max(trun.timestamps)
            if np.min(trun.tdata[variable]) < min_val:
                min_val = np.min(trun.tdata[variable])
            if np.max(trun.tdata[variable]) > max_val:
                max_val = np.max(trun.tdata[variable])
            if trun.tot_timesteps > max_num_ts:
                max_num_ts = trun.tot_timesteps
        num_ts = 2 * max_num_ts
        anim_dt = (max_ts - min_ts) / num_ts

        # Plot initial profiles
        lines = [None] * self.num_sims
        for i, trun in enumerate(self.truns):
            if self.sort_by == 'density':
                lines[i], = ax.plot(trun.xgrid, trun.tdata[variable][0, :],
                                    label=self.line_labels[i], color=self.line_colours[i])
            elif self.sort_by == 'power':
                lines[i], = ax.plot(trun.xgrid, trun.tdata[variable][0, :],
                                    label=self.line_labels[i], color=self.line_colours[i])
        ax.set_ylim([min_val, max_val])
        ax.set_xlabel('x [m]')
        ax.set_ylabel(self.truns[0].get_var_label(variable))
        ax.legend()

        # Animate
        ts_counts = [0] * self.num_sims

        def update_lines(i):
            cur_ts = anim_dt * i
            ax.set_title('{:.1f} $\mu$s'.format(1e6 * cur_ts))
            for j in range(self.num_sims):
                ts_counts[j] = min(bisect.bisect_left(
                    self.truns[j].timestamps, cur_ts), len(self.truns[j].timestamps)-1)
                lines[j].set_data(
                    self.truns[j].xgrid, self.truns[j].tdata[variable][ts_counts[j], :])
            return lines,
        anim = animation.FuncAnimation(
            fig, update_lines, frames=num_ts, interval=interval, blit=False, repeat=True)
        plt.show()

        return anim

    def timeseries(self, variable, cell=-1):

        all_tseries = []
        for trun in self.truns:
            trun.load_tvars(variable)
            all_tseries.append(trun.tdata[variable][:, cell])

        # Set up figure and colour map
        fig, ax = plt.subplots(1)

        for i, trun in enumerate(self.truns):
            ax.plot(trun.timestamps*1e6,
                    all_tseries[i], label=self.line_labels[i], color=self.line_colours[i])

        ax.legend()
        ax.set_xlabel('Time [$\mu$s]')
        ax.set_ylabel(self.truns[0].get_var_label(variable))
        ax.set_title(
            variable + ' at x={:.2f}m'.format(self.truns[0].xgrid[cell]))

    def burnthrough_plot(self):

        iz_degs = [None] * self.num_sims
        for i, trun in enumerate(self.truns):
            trun.load_tvars(['DENSITY', 'NEUTRAL_DENS'])
            iz_degs[i] = np.zeros(trun.tot_timesteps)
            for j, t in enumerate(trun.timestamps):
                avg_dens_plasma = np.sum(
                    trun.tdata['DENSITY'][j, :] * trun.dxc) / trun.connection_length
                avg_dens_neuts = np.sum(
                    trun.tdata['NEUTRAL_DENS'][j, :] * trun.dxc) / trun.connection_length
                iz_degs[i][j] = 100.0*(1.0 - avg_dens_neuts / avg_dens_plasma)

        # Set up figure and colour map
        fig, ax = plt.subplots(1)

        for i, trun in enumerate(self.truns):
            ax.plot(trun.timestamps*1e6,
                    iz_degs[i], label=self.line_labels[i], color=self.line_colours[i])
        ax.set_ylabel('Ionization degree [%]')
        ax.set_xlabel('Time [$\mu$s]')
        ax.set_title('Line-averaged plasma ionization degree')
        ax.legend()

    def flux_rollover_plot(self, interval=0, anim_dt=1, anim_max_ts=None):
        # anim_dt is animation time interval in microseconds
        # anim_max_ts is max timestamp to animate in microseconds

        # Get timestamp limits
        min_ts = np.min(self.truns[0].timestamps)
        max_ts = np.max(self.truns[0].timestamps)
        max_num_ts = self.truns[0].tot_timesteps
        for trun in self.truns:
            if np.min(trun.timestamps) < min_ts:
                min_ts = np.min(trun.timestamps)
            if np.max(trun.timestamps) > max_ts:
                max_ts = np.max(trun.timestamps)
            if trun.tot_timesteps > max_num_ts:
                max_num_ts = trun.tot_timesteps
        anim_dt = 1e-6 * anim_dt
        if anim_max_ts is None:
            anim_timestamps = np.arange(0.0, max_ts, anim_dt)
        else:
            anim_timestamps = np.arange(0.0, 1e-6 * anim_max_ts, anim_dt)

        # Get target fluxes and target temperatures
        T_t = []
        Gamma_t = []
        for i, trun in enumerate(self.truns):
            trun.load_tvars(['TEMPERATURE', 'ION_TEMPERATURE', 'DENSITY'])
            T_t.append(np.zeros(trun.tot_timesteps))
            Gamma_t.append(np.zeros(trun.tot_timesteps))

            for j in range(len(trun.timestamps)):
                T_t[i][j] = 0.5 * (trun.tdata['TEMPERATURE']
                                   [j, -1] + trun.tdata['ION_TEMPERATURE'][j, -1])
                Gamma_t[i][j] = trun.get_target_flux(j)

            # Resample values to animation timestamps
            interp_T_t = interp1d(
                trun.timestamps, T_t[i], bounds_error=False, fill_value=np.nan)
            interp_Gamma_t = interp1d(
                trun.timestamps, Gamma_t[i], bounds_error=False, fill_value=np.nan)
            T_t[i] = interp_T_t(anim_timestamps)
            Gamma_t[i] = interp_Gamma_t(anim_timestamps)
        T_t = np.array(T_t)
        Gamma_t = np.array(Gamma_t)

        # Set up figure
        fig, ax = plt.subplots(1)
        Gamma_t_line, = ax.plot(
            self.avg_densities, [Gamma_t[i][0] for i in range(self.num_sims)], 'x-', color='black')
        ax2 = ax.twinx()
        T_t_line, = ax2.plot(
            self.avg_densities, [T_t[i][0] for i in range(self.num_sims)], 'x-', color='red')
        ax.set_ylabel('Target ion flux [m$^{-2}$s$^{-1}$]')
        ax2.set_ylabel('Target electron temperature [eV]', color='red')
        ax2.tick_params(axis='y', colors='red')
        ax.set_xlabel(r'Line-averaged density [m$^{-3}$]')
        ax2.set_ylim([np.min(T_t[~np.isnan(T_t)]),
                     np.max(T_t[~np.isnan(T_t)])])
        ax.set_ylim([np.min(Gamma_t[~np.isnan(Gamma_t)]),
                    np.max(Gamma_t[~np.isnan(Gamma_t)])])

        # Animate
        def update_lines(i):
            cur_ts = anim_dt * i
            ax.set_title('{:.1f} $\mu$s'.format(1e6 * cur_ts))
            T_t_line.set_data(
                self.avg_densities, T_t[:, i])
            Gamma_t_line.set_data(
                self.avg_densities, Gamma_t[:, i])

        anim = animation.FuncAnimation(
            fig, update_lines, frames=len(anim_timestamps), interval=interval, blit=False, repeat=True)

        return anim

    def q_sh_e_plot(self, interval=0, anim_dt=1, anim_max_ts=None):
        # anim_dt is animation time interval in microseconds
        # anim_max_ts is max timestamp to animate in microseconds

        # Get timestamp limits
        min_ts = np.min(self.truns[0].timestamps)
        max_ts = np.max(self.truns[0].timestamps)
        max_num_ts = self.truns[0].tot_timesteps
        for trun in self.truns:
            if np.min(trun.timestamps) < min_ts:
                min_ts = np.min(trun.timestamps)
            if np.max(trun.timestamps) > max_ts:
                max_ts = np.max(trun.timestamps)
            if trun.tot_timesteps > max_num_ts:
                max_num_ts = trun.tot_timesteps
        anim_dt = 1e-6 * anim_dt
        if anim_max_ts is None:
            anim_timestamps = np.arange(0.0, max_ts, anim_dt)
        else:
            anim_timestamps = np.arange(0.0, 1e-6 * anim_max_ts, anim_dt)

        # Get target heat fluxes
        q_sh = []
        for i, trun in enumerate(self.truns):
            trun.load_tvars(['TEMPERATURE', 'ION_TEMPERATURE',
                            'DENSITY', 'SHEATH_DATA'])
            q_sh.append(np.zeros(trun.tot_timesteps))

            for j in range(len(trun.timestamps)):
                q_sh[i][j] = 1e-6 * trun.q_sh_e(j)
                if q_sh[i][j] == 0.0 and j > 0:
                    if trun.timestamps[j] == trun.timestamps[j-1]:
                        q_sh[i][j] = q_sh[i][j-1]

            # Resample values to animation timestamps
            interp_q_sh = interp1d(
                trun.timestamps, q_sh[i], bounds_error=False, fill_value=np.nan)
            q_sh[i] = interp_q_sh(anim_timestamps)
        q_sh = np.array(q_sh)

        # Set up figure
        fig, ax = plt.subplots(1)
        q_sh_line, = ax.plot(
            self.avg_densities, q_sh[:, 0], 'x-', color='black')
        ax.set_ylabel('Target heat flux [MWm$^{-2}$]')
        ax.set_xlabel(r'Line-averaged density [m$^{-3}$]')
        ax.set_ylim([np.min(q_sh[~np.isnan(q_sh)]),
                    np.max(q_sh[~np.isnan(q_sh)])])

        # Animate
        def update_lines(i):
            cur_ts = anim_dt * i
            ax.set_title('{:.1f} $\mu$s'.format(1e6 * cur_ts))
            q_sh_line.set_data(
                self.avg_densities, q_sh[:, i])

        anim = animation.FuncAnimation(
            fig, update_lines, frames=len(anim_timestamps), interval=interval, blit=False, repeat=True)

        return anim


class SKTComp(SKComp):

    def __init__(self, trundecks, labels):
        SKComp.__init__(self, trundecks, labels)
        self.trundecks = trundecks

    def timeseries(self, variable, cell=-1):

        all_tseries_comp = []
        for trundeck in self.trundecks:
            all_tseries = []
            for trun in trundeck.truns:
                trun.load_tvars(variable)
                all_tseries.append(trun.tdata[variable][:, cell])
            all_tseries_comp.append(all_tseries)

        fig, ax = plt.subplots(1)
        for j, trundeck in enumerate(self.trundecks):
            for i, trun in enumerate(trundeck.truns):
                ax.plot(trun.timestamps*1e6,
                        all_tseries_comp[j][i], linestyle=self.linestyles[j], color=self.line_colours[j][i])

        ax.legend(self.legend_lines, self.legend_labels)
        ax.set_xlabel('Time [$\mu$s]')
        ax.set_ylabel(self.trundecks[0].truns[0].get_var_label(variable))
        ax.set_title(
            variable + ' at x={:.2f}m'.format(self.trundecks[0].truns[0].xgrid[cell]))

    def animate(self, variable, interval=5):

        for trundeck in self.trundecks:
            for trun in trundeck.truns:
                trun.load_tvars(variable)

        fig, ax = plt.subplots(1)

        # Get timestamp and variable limits
        min_ts = np.min(self.trundecks[0].truns[0].timestamps)
        max_ts = np.max(self.trundecks[0].truns[0].timestamps)
        max_num_ts = self.trundecks[0].truns[0].tot_timesteps
        min_val = np.min(self.trundecks[0].truns[0].tdata[variable])
        max_val = np.max(self.trundecks[0].truns[0].tdata[variable])
        for trundeck in self.trundecks:
            for trun in trundeck.truns:
                if np.min(trun.timestamps) < min_ts:
                    min_ts = np.min(trun.timestamps)
                if np.max(trun.timestamps) > max_ts:
                    max_ts = np.max(trun.timestamps)
                if np.min(trun.tdata[variable]) < min_val:
                    min_val = np.min(trun.tdata[variable])
                if np.max(trun.tdata[variable]) > max_val:
                    max_val = np.max(trun.tdata[variable])
                if trun.tot_timesteps > max_num_ts:
                    max_num_ts = trun.tot_timesteps
        num_ts = 2 * max_num_ts
        anim_dt = (max_ts - min_ts) / num_ts

        # Plot initial profiles
        legend_lines = []
        legend_labels = []
        lines = [None] * len(self.trundecks)
        for j, trundeck in enumerate(self.trundecks):
            lines[j] = [None] * trundeck.num_sims
            for i, trun in enumerate(trundeck.truns):

                lines[j][i], = ax.plot(trun.xgrid, trun.tdata[variable][0, :],
                                       linestyle=self.linestyles[j], color=self.line_colours[j][i])

        ax.set_ylim([min_val, max_val])
        ax.set_xlabel('x [m]')
        ax.set_ylabel(trundeck.truns[0].get_var_label(variable))
        ax.legend(self.legend_lines, self.legend_labels)

        # Animate
        ts_counts = [None] * len(self.trundecks)
        for i, trundeck in enumerate(self.trundecks):
            ts_counts[i] = [0] * trundeck.num_sims

        def update_lines(i):
            cur_ts = anim_dt * i
            ax.set_title('{:.1f} $\mu$s'.format(1e6 * cur_ts))
            for k, trundeck in enumerate(self.trundecks):
                for j in range(trundeck.num_sims):
                    ts_counts[k][j] = bisect.bisect_left(
                        trundeck.truns[j].timestamps, cur_ts)
                    lines[k][j].set_data(
                        trundeck.truns[j].xgrid, trundeck.truns[j].tdata[variable][ts_counts[k][j], :])
            return lines,
        anim = animation.FuncAnimation(
            fig, update_lines, frames=num_ts, interval=interval, blit=True, repeat=True)
        plt.show()

        return anim

    def burnthrough_plot(self):

        all_iz_degs = []
        for j, trundeck in enumerate(self.trundecks):
            all_iz_degs.append([None] * trundeck.num_sims)
            for i, trun in enumerate(trundeck.truns):
                trun.load_tvars(['DENSITY', 'NEUTRAL_DENS'])
                all_iz_degs[j][i] = np.zeros(trun.tot_timesteps)
                for k, t in enumerate(trun.timestamps):
                    avg_dens_plasma = np.sum(
                        trun.tdata['DENSITY'][k, :] * trun.dxc) / trun.connection_length
                    avg_dens_neuts = np.sum(
                        trun.tdata['NEUTRAL_DENS'][k, :] * trun.dxc) / trun.connection_length
                    all_iz_degs[j][i][k] = 100.0 * \
                        (1.0 - avg_dens_neuts / avg_dens_plasma)

        # Set up figure and colour map
        fig, ax = plt.subplots(1)

        for j, trundeck in enumerate(self.trundecks):
            for i, trun in enumerate(trundeck.truns):
                ax.plot(trun.timestamps*1e6,
                        all_iz_degs[j][i], linestyle=self.linestyles[j], color=self.line_colours[j][i])
        ax.set_ylabel('Ionization degree [%]')
        ax.set_xlabel('Time [$\mu$s]')
        ax.set_title('Line-averaged plasma ionization degree')
        ax.legend(self.legend_lines, self.legend_labels)

    def targetflux_plot(self):

        targetfluxes = []
        for j, trundeck in enumerate(self.trundecks):
            targetfluxes.append([None] * trundeck.num_sims)
            for i, trun in enumerate(trundeck.truns):
                trun.load_tvars(['DENSITY', 'TEMPERATURE', 'ION_TEMPERATURE'])
                targetfluxes[j][i] = np.zeros(trun.tot_timesteps)
                for k, t in enumerate(trun.timestamps):
                    n_t = trun.tdata['DENSITY'][k][-1] * \
                        trun.tdata['DENSITY'][k][-1] / \
                        trun.tdata['DENSITY'][k][-2]
                    T_et = trun.tdata['TEMPERATURE'][k][-1] / trun.T_norm
                    T_it = trun.tdata['ION_TEMPERATURE'][k][-1] / trun.T_norm
                    c_s = sound_speed(T_et, T_it)
                    targetfluxes[j][i][k] = n_t * c_s * trun.v_th

        # Set up figure and colour map
        fig, ax = plt.subplots(1)

        for j, trundeck in enumerate(self.trundecks):
            for i, trun in enumerate(trundeck.truns):
                ax.plot(trun.timestamps*1e6,
                        targetfluxes[j][i], linestyle=self.linestyles[j], color=self.line_colours[j][i])
        ax.set_ylabel('Target particle flux [m$^{-2}$s$^{-1}$]')
        ax.set_xlabel('Time [$\mu$s]')
        ax.set_title('Target particle flux')
        ax.legend(self.legend_lines, self.legend_labels)

    def flux_rollover_plot(self, interval=0, anim_dt=1, anim_max_ts=None):
        # anim_dt is animation time interval in microseconds
        # anim_max_ts is max timestamp to animate in microseconds

        # Get timestamp limits
        min_ts = np.min(self.trundecks[0].truns[0].timestamps)
        max_ts = np.max(self.trundecks[0].truns[0].timestamps)
        max_num_ts = self.trundecks[0].truns[0].tot_timesteps
        for trundeck in self.trundecks:
            for trun in trundeck.truns:
                if np.min(trun.timestamps) < min_ts:
                    min_ts = np.min(trun.timestamps)
                if np.max(trun.timestamps) > max_ts:
                    max_ts = np.max(trun.timestamps)
                if trun.tot_timesteps > max_num_ts:
                    max_num_ts = trun.tot_timesteps
        anim_dt = 1e-6 * anim_dt
        if anim_max_ts is None:
            anim_timestamps = np.arange(0.0, max_ts, anim_dt)
        else:
            anim_timestamps = np.arange(0.0, 1e-6 * anim_max_ts, anim_dt)

        # Get target fluxes and target temperatures
        T_t = [None] * len(self.trundecks)
        Gamma_t = [None] * len(self.trundecks)
        for k, trundeck in enumerate(self.trundecks):
            T_t[k] = []
            Gamma_t[k] = []
            for i, trun in enumerate(trundeck.truns):
                trun.load_tvars(['TEMPERATURE', 'ION_TEMPERATURE', 'DENSITY'])
                T_t[k].append(np.zeros(trun.tot_timesteps))
                Gamma_t[k].append(np.zeros(trun.tot_timesteps))

                for j in range(len(trun.timestamps)):
                    T_t[k][i][j] = 0.5 * (trun.tdata['TEMPERATURE']
                                          [j, -1] + trun.tdata['ION_TEMPERATURE'][j, -1])
                    Gamma_t[k][i][j] = trun.get_target_flux(j)

                # Resample values to animation timestamps
                interp_T_t = interp1d(
                    trun.timestamps, T_t[k][i], bounds_error=False, fill_value=np.nan)
                interp_Gamma_t = interp1d(
                    trun.timestamps, Gamma_t[k][i], bounds_error=False, fill_value=np.nan)
                T_t[k][i] = interp_T_t(anim_timestamps)
                Gamma_t[k][i] = interp_Gamma_t(anim_timestamps)
            T_t[k] = np.array(T_t[k])
            Gamma_t[k] = np.array(Gamma_t[k])

        # Get limits
        min_T_t = np.min([np.min(T_t[k][~np.isnan(T_t[k])])
                         for k in range(len(self.trundecks))])
        max_T_t = np.max([np.max(T_t[k][~np.isnan(T_t[k])])
                         for k in range(len(self.trundecks))])
        min_Gamma_t = np.min([np.min(Gamma_t[k][~np.isnan(Gamma_t[k])])
                             for k in range(len(self.trundecks))])
        max_Gamma_t = np.max([np.max(Gamma_t[k][~np.isnan(Gamma_t[k])])
                             for k in range(len(self.trundecks))])

        # Set up figure
        fig, ax = plt.subplots(1)
        ax2 = ax.twinx()
        T_t_lines = [None] * len(self.trundecks)
        Gamma_t_lines = [None] * len(self.trundecks)
        for k, trundeck in enumerate(self.trundecks):
            Gamma_t_lines[k], = ax.plot(
                trundeck.avg_densities, Gamma_t[k][:, 0], marker='x', linestyle=self.linestyles[k], color='black', label=self.labels[k])
            T_t_lines[k], = ax2.plot(
                trundeck.avg_densities, T_t[k][:, 0], marker='x', linestyle=self.linestyles[k], color='red')
        ax.set_ylabel('Target ion flux [m$^{-2}$s$^{-1}$]')
        ax2.set_ylabel('Target electron temperature [eV]', color='red')
        ax2.tick_params(axis='y', colors='red')
        ax.set_xlabel(r'Line-averaged density [m$^{-3}$]')
        ax2.set_ylim([min_T_t, max_T_t])
        ax.set_ylim([min_Gamma_t, max_Gamma_t])
        ax.legend()

        # Animate
        def update_lines(i):
            cur_ts = anim_dt * i
            ax.set_title('{:.1f} $\mu$s'.format(1e6 * cur_ts))
            for k, trundeck in enumerate(self.trundecks):
                T_t_lines[k].set_data(
                    trundeck.avg_densities, T_t[k][:, i])
                Gamma_t_lines[k].set_data(
                    trundeck.avg_densities, Gamma_t[k][:, i])

        anim = animation.FuncAnimation(
            fig, update_lines, frames=len(anim_timestamps), interval=interval, blit=False, repeat=True)

        return anim

    def q_sh_e_plot(self, interval=0, anim_dt=1, anim_max_ts=None):
        # anim_dt is animation time interval in microseconds
        # anim_max_ts is max timestamp to animate in microseconds

        # Get timestamp limits
        min_ts = np.min(self.trundecks[0].truns[0].timestamps)
        max_ts = np.max(self.trundecks[0].truns[0].timestamps)
        max_num_ts = self.trundecks[0].truns[0].tot_timesteps
        for trundeck in self.trundecks:
            for trun in trundeck.truns:
                if np.min(trun.timestamps) < min_ts:
                    min_ts = np.min(trun.timestamps)
                if np.max(trun.timestamps) > max_ts:
                    max_ts = np.max(trun.timestamps)
                if trun.tot_timesteps > max_num_ts:
                    max_num_ts = trun.tot_timesteps
        anim_dt = 1e-6 * anim_dt
        if anim_max_ts is None:
            anim_timestamps = np.arange(0.0, max_ts, anim_dt)
        else:
            anim_timestamps = np.arange(0.0, 1e-6 * anim_max_ts, anim_dt)

        # Get target heat fluxes
        q_sh = [None] * len(self.trundecks)
        for k, trundeck in enumerate(self.trundecks):
            q_sh[k] = []
            for i, trun in enumerate(trundeck.truns):
                trun.load_tvars(['TEMPERATURE', 'ION_TEMPERATURE',
                                 'DENSITY', 'SHEATH_DATA'])
                q_sh[k].append(np.zeros(trun.tot_timesteps))

                for j in range(len(trun.timestamps)):
                    q_sh[k][i][j] = 1e-6 * trun.q_sh_e(j)
                    if q_sh[k][i][j] == 0.0 and j > 0:
                        if trun.timestamps[j] == trun.timestamps[j-1]:
                            q_sh[k][i][j] = q_sh[k][i][j-1]

                # Resample values to animation timestamps
                interp_q_sh = interp1d(
                    trun.timestamps, q_sh[k][i], bounds_error=False, fill_value=np.nan)
                q_sh[k][i] = interp_q_sh(anim_timestamps)
            q_sh[k] = np.array(q_sh[k])

        # Get limits
        min_q_sh = np.min([np.min(q_sh[k][~np.isnan(q_sh[k])])
                           for k in range(len(self.trundecks))])
        max_q_sh = np.max([np.max(q_sh[k][~np.isnan(q_sh[k])])
                           for k in range(len(self.trundecks))])

        # Set up figure
        fig, ax = plt.subplots(1)
        q_sh_lines = [None] * len(self.trundecks)
        for k, trundeck in enumerate(self.trundecks):
            q_sh_lines[k], = ax.plot(
                trundeck.avg_densities, q_sh[k][:, 0], marker='x', linestyle=self.linestyles[k], color='black', label=self.labels[k])
        ax.set_ylabel('Target heat flux [MWm$^{-2}$]')
        ax.set_xlabel(r'Line-averaged density [m$^{-3}$]')
        ax.set_ylim([min_q_sh, max_q_sh])
        ax.legend()

        # Animate
        def update_lines(i):
            cur_ts = anim_dt * i
            ax.set_title('{:.1f} $\mu$s'.format(1e6 * cur_ts))
            for k, trundeck in enumerate(self.trundecks):
                q_sh_lines[k].set_data(
                    trundeck.avg_densities, q_sh[k][:, i])

        anim = animation.FuncAnimation(
            fig, update_lines, frames=len(anim_timestamps), interval=interval, blit=False, repeat=True)

        return anim


def SD1D_ion_rate():
    ion_rate = np.array([[1.0395725771549431, 2.2303299924276372e-20],
                         [1.1583146909840112, 6.435648933092225e-20],
                         [1.2532839996830303, 1.491407551599931e-19],
                         [1.371477510316791, 3.6508284169115454e-19],
                         [1.74708783137158, 2.929916365683758e-18],
                         [2.027180240532023, 8.607695520236041e-18],
                         [2.376457608211971, 2.57513598797426e-17],
                         [2.9105532968004244, 8.913716606383189e-17],
                         [3.69761601433685, 2.6645547337371894e-16],
                         [4.668521758212285, 6.879430670462022e-16],
                         [6.4922006912400265, 1.9078735119027417e-15],
                         [8.129964096968664, 3.2911297502631833e-15],
                         [10.205269028911388, 5.272870961018532e-15],
                         [12.510560611509696, 7.430382957245858e-15],
                         [19.609242135163637, 1.3162718010197457e-14],
                         [24.86345876027869, 1.6532942957243723e-14],
                         [25.186037439120295, 1.6523199726930156e-14]])

    return ion_rate


def ADAS_ion_rate():
    adas_temps = np.array([1.000E+00, 2.000E+00, 3.000E+00, 4.000E+00, 5.000E+00, 7.000E+00,
                           1.000E+01, 1.500E+01, 2.000E+01, 3.000E+01, 4.000E+01, 5.000E+01,
                           7.000E+01, 1.000E+02, 1.500E+02, 2.000E+02, 3.000E+02, 4.000E+02,
                           5.000E+02, 7.000E+02, 1.000E+03, 2.000E+03, 5.000E+03, 1.000E+04])
    adas_rates = np.array([7.570E-15, 9.736E-12, 1.168E-10, 4.247E-10, 9.479E-10, 2.474E-09,
                           5.319E-09, 1.006E-08, 1.408E-08, 1.994E-08, 2.374E-08, 2.627E-08,
                           2.917E-08, 3.094E-08, 3.142E-08, 3.094E-08, 2.943E-08, 2.795E-08,
                           2.664E-08, 2.452E-08, 2.218E-08, 1.774E-08, 1.270E-08, 9.686E-09])
    ion_rate = np.concatenate([adas_temps, 1e-6*adas_rates])
    ion_rate = ion_rate.reshape([len(adas_temps), 2], order='F')

    return ion_rate


# amjuel_ion_coeffs_JH = np.array([[-3.292647100524E+01, 1.293481375348E-02, 5.517562508468E-03, -7.853816321645E-04, 1.436128501544E-04, -3.883750282085E-07, -1.489774355194E-06, 1.416361431167E-07, -3.890932078762E-09],
amjuel_ion_coeffs = np.array([[-3.248025330340E+01, -5.440669186583E-02, 9.048888225109E-02, -4.054078993576E-02, 8.976513750477E-03, -1.060334011186E-03, 6.846238436472E-05, -2.242955329604E-06, 2.890437688072E-08],
                              [1.425332391510E+01, -3.594347160760E-02, -2.014729121556E-02, 1.039773615730E-02, -
                               1.771792153042E-03, 1.237467264294E-04, -3.130184159149E-06, -3.051994601527E-08, 1.888148175469E-09],
                              [-6.632235026785E+00, 9.255558353174E-02, -5.580210154625E-03, -5.902218748238E-03,
                               1.295609806553E-03, -1.056721622588E-04, 4.646310029498E-06, -1.479612391848E-07, 2.852251258320E-09],
                              [2.059544135448E+00, -7.562462086943E-02, 1.519595967433E-02, 5.803498098354E-04, -3.527285012725E-04,
                               3.201533740322E-05, -1.835196889733E-06, 9.474014343303E-08, -2.342505583774E-09],
                              [-4.425370331410E-01, 2.882634019199E-02, -7.285771485050E-03, 4.643389885987E-04, 1.145700685235E-06,
                               8.493662724988E-07, -1.001032516512E-08, -1.476839184318E-08, 6.047700368169E-10],
                              [6.309381861496E-02, -5.788686535780E-03, 1.507382955250E-03, -1.201550548662E-04,
                               6.574487543511E-06, -9.678782818849E-07, 5.176265845225E-08, 1.291551676860E-09, -9.685157340473E-11],
                              [-5.620091829261E-03, 6.329105568040E-04, -1.527777697951E-04, 8.270124691336E-06, 3.224101773605E-08,
                               4.377402649057E-08, -2.622921686955E-09, -2.259663431436E-10, 1.161438990709E-11],
                              [2.812016578355E-04, -3.564132950345E-05, 7.222726811078E-06, 1.433018694347E-07, -1.097431215601E-07,
                               7.789031791949E-09, -4.197728680251E-10, 3.032260338723E-11, -8.911076930014E-13],
                              [-6.011143453374E-06, 8.089651265488E-07, -1.186212683668E-07, -2.381080756307E-08, 6.271173694534E-09, -5.483010244930E-10, 3.064611702159E-11, -1.355903284487E-12, 2.935080031599E-14]])
amjuel_rec_coeffs = np.array([[-2.855728479302E+01, 3.488563234375E-02, -2.799644392058E-02, 1.209545317879E-02, -2.436630799820E-03, 2.837893719800E-04, -1.886511169084E-05, 6.752155602894E-07, -1.005893858779E-08],
                              [-7.664042607917E-01, -3.583233366133E-03, -7.452514292790E-03, 2.709299760454E-03, -
                               7.745129766167E-04, 1.142444698207E-04, -9.382783518064E-06, 3.902800099653E-07, -6.387411585521E-09],
                              [-4.930424003280E-03, -3.620245352252E-03, 6.958711963182E-03, -2.139257298118E-03,
                               4.603883706734E-04, -5.991636837395E-05, 4.729262545726E-06, -1.993485395689E-07, 3.352589865190E-09],
                              [-5.386830982777E-03, -9.532840484460E-04, 4.631753807534E-04, -5.371179699661E-04,
                               1.543350502150E-04, -2.257565836876E-05, 1.730782954588E-06, -6.618240780594E-08, 1.013364275013E-09],
                              [-1.626039237665E-04, 1.888048628708E-04, 1.288577690147E-04, -1.634580516353E-05, -
                               9.601036952725E-06, 3.425262385387E-06, -4.077019941998E-07, 2.042041097083E-08, -3.707977721109E-10],
                              [6.080907650243E-06, -1.014890683861E-05, -1.145028889459E-04, 5.942193980802E-05, -
                               1.211851723717E-05, 1.118965496365E-06, -4.275321573501E-08, 3.708616111085E-10, 7.068450112690E-12],
                              [2.101102051942E-05, 2.245676563601E-05, -2.245624273814E-06, -2.944873763540E-06,
                               1.002105099354E-06, -1.291320799814E-07, 7.786155463269E-09, -2.441127783437E-10, 3.773208484020E-12],
                              [-2.770717597683E-06, -4.695982369246E-06, 3.250878872873E-06, -9.387290785993E-07,
                               1.392391630459E-07, -1.139093288575E-08, 5.178505597480E-10, -9.452402157390E-12, -4.672724022059E-14],
                              [1.038235939800E-07, 2.523166611507E-07, -2.145390398476E-07, 7.381435237585E-08, -1.299713684966E-08, 1.265189576423E-09, -6.854203970018E-11, 1.836615031798E-12, -1.640492364811E-14]])
amjuel_ion_E_coeffs = np.array([[-2.497580168306E+01, 1.081653961822E-03, -7.358936044605E-04, 4.122398646951E-04, -1.408153300988E-04, 2.469730836220E-05, -2.212823709798E-06, 9.648139704737E-08, -1.611904413846E-09],
                                [1.004448839974E+01, -3.189474633369E-03, 2.510128351932E-03, -7.707040988954E-04,
                                 1.031309578578E-04, -3.716939423005E-06, -4.249704742353E-07, 4.164960852522E-08, -9.893423877739E-10],
                                [-4.867952931298E+00, -5.852267850690E-03, 2.867458651322E-03, -8.328668093987E-04,
                                 2.056134355492E-04, -3.301570807523E-05, 2.831739755462E-06, -1.164969298033E-07, 1.785440278790E-09],
                                [1.689422238067E+00, 7.744372210287E-03, -3.087364236497E-03, 4.707676288420E-04, -5.508611815406E-05,
                                 7.305867762241E-06, -6.000115718138E-07, 2.045211951761E-08, -1.790312871690E-10],
                                [-4.103532320100E-01, -3.622291213236E-03, 1.327415215304E-03, -1.424078519508E-04,
                                 3.307339563081E-06, 5.256679519499E-09, 7.597020291557E-10, 1.799505288362E-09, -9.280890205774E-11],
                                [6.469718387357E-02, 8.268567898126E-04, -2.830939623802E-04, 2.411848024960E-05,
                                 5.707984861100E-07, -1.016945693300E-07, 3.517154874443E-09, -4.453195673947E-10, 2.002478264932E-11],
                                [-6.215861314764E-03, -9.836595524255E-05, 3.017296919092E-05, -1.474253805845E-06, -
                                 2.397868837417E-07, 1.518743025531E-08, 4.149084521319E-10, -6.803200444549E-12, -1.151855939531E-12],
                                [3.289809895460E-04, 5.845697922558E-06, -1.479323780613E-06, -4.633029022577E-08,
                                 3.337390374041E-08, -1.770252084837E-09, -5.289806153651E-11, 3.864394776250E-12, -8.694978774411E-15],
                                [-7.335808238917E-06, -1.367574486885E-07, 2.423236476442E-08, 5.733871119707E-09, -1.512777532459E-09, 8.733801272834E-11, 7.196798841269E-13, -1.441033650378E-13, 1.734769090475E-15]])
amjuel_rec_E_coeffs = np.array([[-2.592450349909E+01, 1.222097271874E-02, 4.278499401907E-05, 1.943967743593E-03, -7.123474602102E-04, 1.303523395892E-04, -1.186560752561E-05, 5.334455630031E-07, -9.349857887253E-09],
                                [-7.290670236493E-01, -1.540323930666E-02, -3.406093779190E-03, 1.532243431817E-03, -
                                 4.658423772784E-04, 5.972448753445E-05, -4.070843294052E-06, 1.378709880644E-07, -1.818079729166E-09],
                                [2.363925869096E-02, 1.164453346305E-02, -5.845209334594E-03, 2.854145868307E-03, -5.077485291132E-04,
                                 4.211106637742E-05, -1.251436618314E-06, -1.626555745259E-08, 1.073458810743E-09],
                                [3.645333930947E-03, -1.005820792983E-03, 6.956352274249E-04, -9.305056373739E-04,
                                 2.584896294384E-04, -3.294643898894E-05, 2.112924018518E-06, -6.544682842175E-08, 7.810293075700E-10],
                                [1.594184648757E-03, -1.582238007548E-05, 4.073695619272E-04, -9.379169243859E-05, 1.490890502214E-06,
                                 2.245292872209E-06, -3.150901014513E-07, 1.631965635818E-08, -2.984093025695E-10],
                                [-1.216668033378E-03, -3.503070140126E-04, 1.043500296633E-04, 9.536162767321E-06, -
                                 6.908681884097E-06, 8.232019008169E-07, -2.905331051259E-08, -3.169038517749E-10, 2.442765766167E-11],
                                [2.376115895241E-04, 1.172709777146E-04, -6.695182045674E-05, 1.188184006210E-05, -
                                 4.381514364966E-07, -6.936267173079E-08, 6.592249255001E-09, -1.778887958831E-10, 1.160762106747E-12],
                                [-1.930977636766E-05, -1.318401491304E-05, 8.848025453481E-06, -2.072370711390E-06,
                                 2.055919993599E-07, -7.489632654212E-09, -7.073797030749E-11, 1.047087505147E-11, -1.877446271350E-13],
                                [5.599257775146E-07, 4.977823319311E-07, -3.615013823092E-07, 9.466989306497E-08, -1.146485227699E-08, 6.772338917155E-10, -1.776496344763E-11, 7.199195061382E-14, 3.929300283002E-15]])


def amjuel_rate(process, moment, n_in, T_in):
    # n in m^-3, T in eV
    dens_param = n_in / (1e8 * 1e6)

    if process == 'ionisation':
        if moment == 0:
            amjuel_coeffs = amjuel_ion_coeffs
        if moment == 2:
            amjuel_coeffs = amjuel_ion_E_coeffs
    elif process == 'recombination':
        if moment == 0:
            amjuel_coeffs = amjuel_rec_coeffs
        if moment == 2:
            amjuel_coeffs = amjuel_rec_E_coeffs

    num_n_coeffs = amjuel_coeffs.shape[0]
    num_T_coeffs = amjuel_coeffs.shape[1]

    ln_sigmav = 0.0
    ln_n = np.log(dens_param)
    ln_T = np.log(T_in)
    for m in range(num_n_coeffs):
        for n in range(num_T_coeffs):
            ln_sigmav += amjuel_coeffs[n, m] * (ln_n ** m) * (ln_T ** n)

    sigmav = 1e-6 * np.exp(ln_sigmav)

    return sigmav


def maxwellian(T, n, vgrid=None):

    if vgrid is None:
        velocities = np.arange(0.00001, 10, 1. / 1000.)
    else:
        velocities = vgrid

    f = np.zeros(len(velocities))
    for i, v in enumerate(velocities):
        f[i] = (n * (np.pi * T) ** (-3/2) * np.exp(-(v**2) / T))

    return f


def bimaxwellian(T1, T2, n1, n2, vgrid=None):

    if vgrid is None:
        velocities = np.arange(0.00001, 10, 1. / 1000.)
    else:
        velocities = vgrid

    f = np.zeros(len(velocities))
    for i, v in enumerate(velocities):
        f[i] = (n1 * (np.pi * T1) ** (-3/2) * np.exp(-(v**2) / T1)) + \
            (n2 * (np.pi * T2) ** (-3/2) * np.exp(-(v**2) / T2))

    return f


def lambda_ei(n, T, T_0, n_0, Z_0):
    if T * T_0 < 10.00 * Z_0 ** 2:
        return 23.00 - np.log(np.sqrt(n * n_0 * 1.00E-6) * Z_0 * (T * T_0) ** (-3.00/2.00))
    else:
        return 24.00 - np.log(np.sqrt(n * n_0 * 1.00E-6) / (T * T_0))


def lambda_ei_new(n, T, T_0, n_0, Z_0):
    if T * T_0 < 10.00 * Z_0 ** 2:
        return 16.10 - np.log(np.sqrt(n * n_0 * 1.00E-6) * Z_0 * (T * T_0) ** (-3.00/2.00))
    else:
        return 17.10 - np.log(np.sqrt(n * n_0 * 1.00E-6) / (T * T_0))


def sound_speed(T_e, T_i=None):
    c_s = np.sqrt(0.5*((T_e + T_i) * (el_mass / ion_mass)))
    return c_s


def legendre_coeff(l, x):
    if l == 0:
        return 1
    elif l == 1:
        return x
    elif l == 2:
        return 0.5 * ((3*x**2) - 1)
    elif l == 3:
        return 0.5 * ((5.0 * (x**3)) - (3.0*x))


def bisection_left(array, val):
    left_idx = len(array)-1
    for i, x in enumerate(array):
        if val > x:
            left_idx = i
            return left_idx
    return left_idx


def interp_f(f, v, th, vgrid, thgrid):

    v_idx = min(bisect.bisect_left(vgrid, v), len(vgrid)-2)
    th_idx = min(bisect.bisect_left(thgrid, th), len(thgrid)-2)
    # v_idx = min(bisection_left(vgrid, v), len(vgrid)-2)
    # th_idx = min(bisection_left(thgrid, th), len(thgrid)-2)

    dv1 = v - vgrid[v_idx]
    dv2 = vgrid[v_idx+1] - v

    dth1 = th - thgrid[th_idx]
    dth2 = thgrid[th_idx+1] - th

    f1 = (dv2 * f[v_idx, th_idx] + dv1 * f[v_idx+1, th_idx]) / (dv1 + dv2)
    f2 = (dv2 * f[v_idx, th_idx+1] + dv1 * f[v_idx+1, th_idx+1]) / (dv1 + dv2)

    f_val = (dth2 * f1 + dth1 * f2) / (dth1 + dth2)

    return f_val

    # if dv1 > dv2:
    #   v_idx += 1
    # if dth1 > dth2:
    #   th_idx += 1
    # return f[v_idx,th_idx]
