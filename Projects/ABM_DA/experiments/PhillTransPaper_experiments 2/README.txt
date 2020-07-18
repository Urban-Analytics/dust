=================================================================================================================
* gcs_final_real_data/: Folder with real data organized by frame and with an activation.dat file with initial parameters.

* stationsim_gcs_model.py: python3 file with the StationSim model (last version).

* particle_filter_gcs.py: python3 file with the Particle Filter code adapted to use external data.

* run_pf_exp#.py: python3 files to run the experiments for the PhillTransPaper, where # = 1, 2, 3, 4.



=================================================================================================================

Experiment 1: without the DA, giving all parameters, from real data to the agents.
              Here, the number of particles defines the number of runs used to determine the mean distance result.
--------------------------
Script: run_pf_exp1.py
Parameters:
            'do_resample': False, # False for experiments without D.A.
            'external_info': ['gcs_final_real_data/', True, True]}  # [Real data dir, Use external velocit?, Use external gate_out?]

=================================================================================================================

Experiment 2: without the DA, giving only time activation and gate_in, from real data to the agents.
              Here, the number of particles defines the number of runs used to determine the mean distance result.
--------------------------
Script: run_pf_exp2.py
Parameters:
            'do_resample': False, # False for experiments without D.A.
            'external_info': ['gcs_final_real_data/', False, False]}  # [Real data dir, Use external velocit?, Use external gate_out?]

=================================================================================================================

Experiment 3: with DA, giving all parameters, from real data to the agents.
--------------------------
Script: run_pf_exp3.py
Parameters:
            'do_resample': True, # True for experiments with D.A.
            'external_info': ['gcs_final_real_data/', True, True]}  # [Real data dir, Use external velocit?, Use external gate_out?]

=================================================================================================================

Experiment 4: with DA, giving only time activation and gate_in, from real data to the agents.
-------------------------
Script: run_pf_exp4.py
Parameters:
            'do_resample': True, # True for experiments with D.A.
            'external_info': ['gcs_final_real_data/', False, False]}  # [Real data dir, Use external velocit?, Use external gate_out?]

=================================================================================================================