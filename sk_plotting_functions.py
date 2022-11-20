import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.sankey import Sankey

el_mass = 9.10938e-31
ion_mass = 2 * 1.67262e-27
epsilon_0 = 8.854188E-12
el_charge = 1.602189E-19
boltzmann_k = 1.38064852E-23
bohr_radius = 5.291772e-11


class SKComp:
    def __init__(self, rundeck1, rundeck2, label1='Fluid', label2='Kinetic'):
        self.rundeck1 = rundeck1
        self.rundeck2 = rundeck2
        self.label1 = label1
        self.label2 = label2

    def flux_rollover_plot(self, sort_by='density'):
        if sort_by == 'density':
            self.rundeck1.sort_by_density()
            self.rundeck2.sort_by_density()
            self.upstream_densities1, self.avg_densities1 = self.rundeck1.get_densities()
            self.upstream_densities2, self.avg_densities2 = self.rundeck2.get_densities()
            self.target_fluxes1, self.target_temps1 = self.rundeck1.get_target_conditions()
            self.target_fluxes2, self.target_temps2 = self.rundeck2.get_target_conditions()

            fig, ax = plt.subplots(1)
            ax.plot(self.avg_densities1, self.target_temps1, '-', color='red')
            ax.plot(self.avg_densities2, self.target_temps2, '--', color='red')
            ax2 = ax.twinx()
            ax2.plot(self.avg_densities1, self.target_fluxes1,
                     '-', color='black', label=self.label1)
            ax2.plot(self.avg_densities2, self.target_fluxes2,
                     '--', color='black', label=self.label2)
            ax2.legend()
            ax2.set_ylabel('Target ion flux [a.u.]')
            ax.set_ylabel('Target plasma temperature [eV]', color='red')
            ax.tick_params(axis='y', colors='red')
            ax.set_xlabel(r'Line-averaged density [$\times 10 ^{19}$m$^{-3}$]')

            fig.tight_layout()

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
                densities1[i] = r.avg_density
                n_e_boundary1[i] = r.get_target_density() * r.n_norm
            for i, r in enumerate(self.rundeck2.runs):
                densities2[i] = r.avg_density
                n_e_boundary2[i] = r.get_target_density() * r.n_norm

            fig, ax = plt.subplots(1)
            ax.plot(densities1, n_e_boundary1, '-', label=self.label1)
            ax.plot(densities2, n_e_boundary2, '--', label=self.label2)
            ax.set_xlabel('Line-averaged density [m$^{-3}$]')
            ax.set_ylabel('Target density [m$^{-3}$]')
            ax.legend()

        elif variable == 'TEMPERATURE':
            T_e_boundary1 = np.zeros(len(self.rundeck1.runs))
            densities1 = np.zeros(len(self.rundeck1.runs))
            T_e_boundary2 = np.zeros(len(self.rundeck2.runs))
            densities2 = np.zeros(len(self.rundeck2.runs))
            for i, r in enumerate(self.rundeck1.runs):
                densities1[i] = r.avg_density
                T_e_boundary1[i] = r.data['TEMPERATURE'][-1] * r.T_norm
            for i, r in enumerate(self.rundeck2.runs):
                densities2[i] = r.avg_density
                T_e_boundary2[i] = r.data['TEMPERATURE'][-1] * r.T_norm

            fig, ax = plt.subplots(1)
            ax.plot(densities1, T_e_boundary1, '-', label=self.label1)
            ax.plot(densities2, T_e_boundary2, '--', label=self.label2)
            ax.set_xlabel('Line-averaged density [m$^{-3}$]')
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
            self.rundeck1.sort_by_power()
            self.rundeck2.sort_by_power()
        elif sort_by == 'density':
            self.rundeck1.sort_by_density()
            self.rundeck2.sort_by_density()

        fig, ax = plt.subplots(1)
        cmap = plt.cm.get_cmap('plasma')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(self.rundeck1.runs[0].avg_density, self.rundeck2.runs[0].avg_density), vmax=1.25*max(
            self.rundeck1.runs[-1].avg_density, self.rundeck2.runs[-1].avg_density)))

        legend_lines = []
        legend_labels = []
        for r in self.rundeck1.runs:
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
            if sort_by == 'density':
                profile_label = r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                    r.avg_density/1e19) + r'$\times 10^{19}$m$^{-3}$'
            elif sort_by == 'power':
                profile_label = str(r.input_power)

            legend_lines.append(Line2D([0],[0], color=sm.to_rgba(r.avg_density)))
            legend_labels.append(profile_label)

            ax.plot(r.xgrid[cells[0]:cells[1]], plot_data[cells[0]:cells[1]],
                    label=profile_label, color=sm.to_rgba(r.avg_density))

        for r in self.rundeck2.runs:
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
            # if sort_by == 'density':
            #     profile_label = r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(r.avg_density/1e19) + r'$\times 10^{19}$m$^{-3}$'
            # elif sort_by == 'power':
            #     profile_label = str(r.input_power)
            ax.plot(r.xgrid[cells[0]:cells[1]], plot_data[cells[0]                    :cells[1]], '--', color=sm.to_rgba(r.avg_density))

        legend_lines.append(Line2D([0], [0], linestyle='-', color='black'))
        legend_lines.append(Line2D([0], [0], linestyle='-', color='black'))
        legend_labels.append(self.label1)
        legend_labels.append(self.label2)

        ax.set_title(variable)
        ax.legend(legend_lines, legend_labels)
        ax.set_ylabel(y_label)
        ax.set_xlabel('x [m]')
        fig.tight_layout()


class SKRundeck:
    def __init__(self, rundeck_dir, identifier='Output_'):
        self.dir = rundeck_dir
        self.load_rundeck(identifier)

    def load_rundeck(self, identifier):
        simulation_folders = [r for r in os.listdir(
            self.dir) if identifier in r and os.path.isdir(os.path.join(self.dir, r))]
        self.runs = []
        for s in simulation_folders:
            run_numbers = sorted([rn for rn in os.listdir(
                os.path.join(self.dir, s)) if 'Run_' in rn])
            run_number = run_numbers[-1]
            self.runs.append(SKRun(os.path.join(self.dir, s, run_number)))

    def get_densities(self):
        upstream_densities = [None] * len(self.runs)
        avg_densities = [None] * len(self.runs)
        for i, run in enumerate(self.runs):
            upstream_densities[i] = run.upstream_density
            avg_densities[i] = run.avg_density
        return upstream_densities, avg_densities

    def get_target_conditions(self):
        target_fluxes = [None] * len(self.runs)
        target_temps = [None] * len(self.runs)
        for i, run in enumerate(self.runs):
            target_fluxes[i], target_temps[i] = run.get_target_conditions()
        return target_fluxes, target_temps

    def profile(self, variable, sort_by='density', cells=(0, None)):
        if sort_by == 'power':
            self.sort_by_power()
        elif sort_by == 'density':
            self.sort_by_density()

        fig, ax = plt.subplots(1)
        cmap = plt.cm.get_cmap('plasma')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
            vmin=self.runs[0].upstream_density, vmax=self.runs[-1].upstream_density))

        for r in self.runs:
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
            if sort_by == 'density':
                profile_label = r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                    r.avg_density/1e19) + r'$\times 10^{19}$m$^{-3}$'
            elif sort_by == 'power':
                profile_label = str(r.input_power)
            ax.plot(r.xgrid[cells[0]:cells[1]], plot_data[cells[0]:cells[1]],
                    label=profile_label, color=sm.to_rgba(r.upstream_density))
        ax.legend()
        ax.set_ylabel(y_label)
        ax.set_xlabel('x [m]')
        fig.tight_layout()

    def flux_rollover_plot(self, sort_by='density'):
        if sort_by == 'density':
            self.sort_by_density()
            self.upstream_densities, self.avg_densities = self.get_densities()
            self.target_fluxes, self.target_temps = self.get_target_conditions()

            fig, ax = plt.subplots(1)
            ax.plot(self.avg_densities, self.target_fluxes, '-', color='black')
            ax2 = ax.twinx()
            ax2.plot(self.avg_densities, self.target_temps, '-', color='red')
            ax.set_ylabel('Target ion flux [a.u.]')
            ax2.set_ylabel('Target plasma temperature [eV]', color='red')
            ax2.tick_params(axis='y', colors='red')
            ax.set_xlabel(r'Line-averaged density [$\times 10 ^{19}$m$^{-3}$]')
            fig.tight_layout()
        elif sort_by == 'power':
            self.sort_by_power()
            self.target_fluxes, self.target_temps = self.get_target_conditions()

            fig, ax = plt.subplots(1)
            ax.plot(self.input_powers, self.target_fluxes, '-', color='black')
            ax2 = ax.twinx()
            ax2.plot(self.input_powers, self.target_temps, '-', color='red')
            ax.set_ylabel('Target ion flux [a.u.]')
            ax2.set_ylabel('Target plasma temperature [eV]', color='red')
            ax2.tick_params(axis='y', colors='red')
            ax.set_xlabel(r'Input power [MWm$^{-2}$]')
            fig.tight_layout()

    def pressure_ratio_plot(self, sort_by='density'):
        if sort_by == 'density':
            self.sort_by_density()

            upstream_pressure = np.zeros(len(self.runs))
            target_pressure = np.zeros(len(self.runs))
            self.upstream_densities, self.avg_densities = self.get_densities()
            for i, r in enumerate(self.runs):
                r.calc_pressure()
                plasma_pressure = r.el_pressure_stat + r.el_pressure_dyn + \
                    r.ion_pressure_stat + r.ion_pressure_dyn
                upstream_pressure[i] = plasma_pressure[0]
                target_pressure[i] = plasma_pressure[-1]
            fig, ax = plt.subplots()
            ax.plot(self.avg_densities, 2*target_pressure/upstream_pressure)
            ax.set_xlabel('Line-averaged density [m$^{-3}$]')
            ax.set_ylabel('Plasma pressure ratio ($2P_t/P_u$)')

    def energy_discrepancy_plot(self, sort_by='density'):
        if sort_by == 'power':
            self.sort_by_power()
        elif sort_by == 'density':
            self.sort_by_density()

        if self.runs[0].neut_mom_eqn is False and self.runs[0].ion_temp_eqn is False:
            discrepancies = np.zeros(len(self.runs))
            input_powers = np.zeros(len(self.runs))
            for i, run in enumerate(self.runs):
                input_powers[i], q_E, q_sh_e, q_sh_i, q_sh_n, q_recyc_n, q_n_e, q_ion_i, q_rec_i, q_rad, q_ie, q_cx = run.get_energy_channels()
                discrepancies[i] = (
                    input_powers[i] - (q_sh_e + q_n_e + q_rad + q_E))

            fix, ax = plt.subplots(1)
            if sort_by == 'power':
                ax.plot(input_powers, discrepancies)
                ax.set_ylabel('Discrepancy [MW$m^{-2}$]')
                ax.set_xlabel('Input power [MW$m^{-2}$]')
                ax.set_title(r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                    self.runs[0].avg_density/1e19) + r'$\times 10^{19}$m$^{-3}$', loc='left')
            elif sort_by == 'density':
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
            if sort_by == 'power':
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
            elif sort_by == 'density':
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
            if sort_by == 'power':
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
            elif sort_by == 'density':
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

    def energy_channels_plot(self, sort_by='density'):

        if sort_by == 'power':
            self.sort_by_power()
        elif sort_by == 'density':
            self.sort_by_density()

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
            if sort_by == 'density':
                self.sort_by_density()
                ax.stackplot(self.densities, q_rad, q_n_e, q_E, q_sh_e)
                ax.set_xlabel('Line-averaged density [m$^{-3}$]')
            elif sort_by == 'power':
                ax.stackplot(input_powers, q_rad, q_n_e, q_E, q_sh_e)
                ax.set_xlabel('Input power [MWm$^{-2}$]')

            ax.legend(['Radiated power', 'Given to neutrals',
                      'E field', 'Sheath'], loc='upper left')
            ax.set_ylabel('Heat flux [MWm$^{-2}$]')

        if self.runs[0].neut_mom_eqn is False and self.runs[0].ion_temp_eqn is True:
            # Diffusive neutrals, with T_i equation

            # fig, ax = plt.subplots(3,1, figsize=(10,20))
            fig, ax = plt.subplots(1)

            if sort_by == 'density':
                self.sort_by_density()
                ax.stackplot(self.densities, q_rad, q_rec_i -
                             q_ion_i, q_n_e, q_sh_i, q_sh_e)
                ax.set_xlabel('Line-averaged density [m$^{-3}$]')

            if sort_by == 'power':
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
                densities[i] = r.avg_density
                n_e_boundary[i] = r.get_target_density() * r.n_norm

            fig, ax = plt.subplots(1)
            ax.plot(densities, n_e_boundary)
            ax.set_xlabel('Line-averaged density [m$^{-3}$]')
            ax.set_ylabel('Target density [m$^{-3}$]')
        elif variable == 'TEMPERATURE':
            T_e_boundary = np.zeros(len(self.runs))
            densities = np.zeros(len(self.runs))
            for i, r in enumerate(self.runs):
                densities[i] = r.avg_density
                T_e_boundary[i] = r.data['TEMPERATURE'][-1] * r.T_norm

            fig, ax = plt.subplots(1)
            ax.plot(densities, T_e_boundary)
            ax.set_xlabel('Line-averaged density [m$^{-3}$]')
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


class SKRun:
    def __init__(self, run_dir=None):
        if run_dir is None:
            self.dir = '/Users/dpower/Documents/01 - PhD/01 - Code/01 - SOL-KiT'
        else:
            self.dir = run_dir
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
            self.neutral_density = self.integrate_x(
                np.sum(self.data['NEUTRAL_DENS'], 1)*self.n_norm) / self.connection_length
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
                        var_files = sorted(os.listdir(var_dir))
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
            N_t = int(grid_input_data[1].split()[2])
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
            self.ion_pressure_dyn = 2.0 * \
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
            ax.plot(self.xgrid[cells[0]:cells[1]], plasma_p[cells[0]:cells[1]], '-', color='blue', label='Static plasma')
            ax.plot(self.xgrid[cells[0]:cells[1]], plasma_dp[cells[0]:cells[1]], '-', color='red', label='Dynamic plasma')
            ax.plot(self.xgrid[cells[0]:cells[1]], self.neut_pressure_stat[cells[0]:cells[1]], '--', color='blue', label='Static neutral')
            ax.plot(self.xgrid[cells[0]:cells[1]], self.neut_pressure_dyn[cells[0]:cells[1]], '--', color='red', label='Dynamic neutral')
            ax.plot(self.xgrid[cells[0]:cells[1]], tot_p[cells[0]:cells[1]], '-', color='black', label='Total')
            ax.set_title(r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                self.avg_density/1e19) + r'$\times 10^{19}$m$^{-3}$')
            ax.legend()
        if option == 2:
            el_p = self.el_pressure_stat + self.el_pressure_dyn
            ion_p = self.ion_pressure_stat + self.ion_pressure_dyn
            neut_p = self.neut_pressure_stat + self.neut_pressure_dyn
            tot_p = el_p + ion_p + neut_p
            fig, ax = plt.subplots(1)
            ax.plot(self.xgrid[cells[0]:cells[1]], el_p[cells[0]:cells[1]], '-', color='red', label='Electrons')
            ax.plot(self.xgrid[cells[0]:cells[1]], ion_p[cells[0]
                    :cells[1]], '-', color='blue', label='Ions')
            ax.plot(self.xgrid[cells[0]:cells[1]], neut_p[cells[0]
                    :cells[1]], '-', color='green', label='Neutrals')
            ax.plot(self.xgrid[cells[0]:cells[1]], tot_p[cells[0]
                    :cells[1]], '-', color='black', label='Total')
            ax.set_title(r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                self.avg_density/1e19) + r'$\times 10^{19}$m$^{-3}$')
            ax.legend()
        if option == 3:
            plasma_p = self.el_pressure_stat + self.ion_pressure_stat
            plasma_dp = self.el_pressure_dyn + self.ion_pressure_dyn
            tot_p = plasma_p + plasma_dp + self.neut_pressure_dyn + self.neut_pressure_stat
            fig, ax = plt.subplots(1)
            ax.plot(self.xgrid[cells[0]:cells[1]], self.el_pressure_stat[cells[0]:cells[1]], '-', color='red', label='Electrons')
            ax.plot(self.xgrid[cells[0]:cells[1]],
                    self.el_pressure_dyn[cells[0]:cells[1]], '--', color='red')
            ax.plot(self.xgrid[cells[0]:cells[1]], self.ion_pressure_stat[cells[0]                    :cells[1]], '-', color='blue', label='Ions')
            ax.plot(self.xgrid[cells[0]:cells[1]],
                    self.ion_pressure_dyn[cells[0]:cells[1]], '--', color='blue')
            ax.plot(self.xgrid[cells[0]:cells[1]], self.neut_pressure_stat[cells[0]:cells[1]], '-', color='green', label='Neutrals')
            ax.plot(self.xgrid[cells[0]:cells[1]],
                    self.neut_pressure_dyn[cells[0]:cells[1]], '--', color='green')
            ax.plot(self.xgrid[cells[0]:cells[1]], tot_p[cells[0]
                    :cells[1]], '-', color='black', label='Total')
            ax.set_title(r'$\langle n_{heavy} \rangle$' + ' = {:1.1f}'.format(
                self.avg_density/1e19) + r'$\times 10^{19}$m$^{-3}$')
            ax.legend()
        if option == 4:
            plasma_p = self.el_pressure_stat + self.ion_pressure_stat + \
                self.el_pressure_dyn + self.ion_pressure_dyn
            neut_p = self.neut_pressure_dyn + self.neut_pressure_stat
            tot_p = plasma_p + neut_p
            fig, ax = plt.subplots(1)
            ax.plot(self.xgrid[cells[0]:cells[1]], plasma_p[cells[0]:cells[1]], '-', color='red', label='Plasma')
            ax.plot(self.xgrid[cells[0]:cells[1]], neut_p[cells[0]                    :cells[1]], '-', color='green', label='Neutrals')
            ax.plot(self.xgrid[cells[0]:cells[1]], tot_p[cells[0]
                    :cells[1]], '-', color='black', label='Total')
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
        else:
            var_norm = 1.0
        return var_norm

    def profile(self, variable, cells=(0, None)):

        fig, ax = plt.subplots()

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
        target_temp = 0.5*(T_i_boundary + T_e_boundary) * self.T_norm
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
            self.load_vars('RAD_REC_E_RATE')
            q_rad_deex = self.integrate_x(self.data['RAD_REC_E_RATE'])
        except:
            q_rad_deex = 0
        try:
            self.load_vars('RAD_DEEX_E_RATE')
            q_rad_rec = self.integrate_x(self.data['RAD_DEEX_E_RATE'])
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
        gamma_e = 2.0 - 0.5 * \
            np.log(2.0*np.pi*(2.0)
                   * (el_mass/(1.0*ion_mass)))

        flux_from_T_eqn = (el_charge * self.T_norm * self.v_th) * (((2/3) * self.data['TEMPERATURE'][-1] * c_s) + (
            (2/(3*self.data['DENSITY'][-1])) * (gamma_e-2.5) * T_e_boundary * c_s * n_e_boundary))

        flux_from_n_eqn = (self.v_th * self.n_norm) * c_s * n_e_boundary

        q_sh_e = (self.n_norm * self.data['DENSITY'][-1]) * ((3/2) + (np.sqrt(el_mass/ion_mass) * (
            self.data['FLOW_VEL_X'][-1] / np.sqrt(self.data['TEMPERATURE'][-1])))) * flux_from_T_eqn
        q_sh_e += (el_charge * self.T_norm * (3/2) *
                   self.data['TEMPERATURE'][-1] * flux_from_n_eqn)

        # E advection correction
        self.load_vars('E_FIELD_X')
        q_sh_e -= (el_charge * self.T_norm * self.n_norm / self.t_norm) * \
            (self.data['E_FIELD_X'][-1] + 0.5*(self.data['E_FIELD_X'][-1] - self.data['E_FIELD_X'][-2])) * self.data['DENSITY'][-1] * \
            self.data['FLOW_VEL_X'][-1] * self.dxc[-1]

        # Pressure gradient correction
        q_sh_e -= 0.5*el_charge * self.T_norm * self.v_th * self.n_norm * (self.data['DENSITY'][-1] / n_e_boundary) * \
            self.data['FLOW_VEL_X'][-1] * self.data['TEMPERATURE'][-1] * \
            (self.data['DENSITY'][-1] - n_e_boundary)

        return q_sh_e

    def q_sh_i(self):
        T_e_boundary = self.data['TEMPERATURE'][-1]
        T_i_boundary = self.data['ION_TEMPERATURE'][-1]
        c_s = sound_speed(T_e_boundary, T_i_boundary)
        n_e_boundary = self.data['DENSITY'][-1] * \
            self.data['DENSITY'][-1] / self.data['DENSITY'][-2]
        gamma_e = 2.0 - 0.5 * \
            np.log(2.0*np.pi*(2.0)
                   * (el_mass/(1.0*ion_mass)))

        q_sh_i = (el_charge * self.T_norm * self.v_th * self.n_norm) * \
            self.data['ION_DENS'][-1] * \
            self.data['ION_TEMPERATURE'][-1] * c_s

        q_sh_i += (el_charge * self.T_norm * self.v_th * self.n_norm) * \
            1.5 * self.data['ION_TEMPERATURE'][-1] * c_s * n_e_boundary

        q_sh_i += (self.v_th ** 3) * self.n_norm * 0.5 * ion_mass * \
            (self.data['ION_VEL'][-1] ** 2) * c_s * n_e_boundary

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

        # # E-field advection correction
        # self.load_vars('E_FIELD_X')
        # q_sh_i += (el_charge * self.T_norm * self.n_norm / self.t_norm) * \
        #     (self.data['E_FIELD_X'][-1] + 0.5*(self.data['E_FIELD_X'][-1] - self.data['E_FIELD_X'][-2])) * self.data['DENSITY'][-1] * \
        #     self.data['FLOW_VEL_X'][-1] * self.dxc[-1]

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
                self.data['TEMPERATURE'][-1]/self.data['TEMPERATURE'][-1]))*(el_mass/(1.0*ion_mass)))
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


def maxwellian(T, n, vgrid=None):

    if vgrid is None:
        velocities = np.arange(0.00001, 10, 1. / 1000.)
    else:
        velocities = vgrid

    f = np.zeros(len(velocities))
    for i, v in enumerate(velocities):
        f[i] = (n * (np.pi * T) ** (-3/2) * np.exp(-(v**2) / T))

    return f


def lambda_ei(n, T, T_0, n_0, Z_0):
    if T * T_0 < 10.00 * Z_0 ** 2:
        return 23.00 - np.log(np.sqrt(n * n_0 * 1.00E-6) * Z_0 * (T * T_0) ** (-3.00/2.00))
    else:
        return 24.00 - np.log(np.sqrt(n * n_0 * 1.00E-6) / (T * T_0))


def sound_speed(T_e, T_i=None):
    c_s = np.sqrt(0.5*((T_e + T_i) * (el_mass / ion_mass)))
    return c_s
