function default_KiD_config()
    return Dict(
        # Mositure model choice: EquilibriumMoisture, NonEquilibriumMoisture, CloudyMoisture, MoistureP3
        "moisture_choice" => "EquilibriumMoisture",
        
        # Precipitation model choice: NoPrecipitation, Precipitation0M, Precipitation1M, Precipitation2M, CloudyPrecip, PrecipitationP3
        "precipitation_choice" => "Precipitation1M",

        # Number of moments to use for CloudyPrecip (ignored otherwise)
        "num_moments" => 6,

        # Rain formation scheme choice: CliMA_1M, KK2000, B1994, TC1980, LD2004, VarTimeScaleAcnv for Precipitation1M; and SB2006, SB2006NL for Precipitation2M
        "rain_formation_scheme_choice" => "CliMA_1M",

        # Sedimentation scheme choice: CliMA_1M, Chen2022 for Precipitation1M; and Chen2022, SB2006 for Precipitation2M
        "sedimentation_scheme_choice" => "CliMA_1M",

        # Prescribed number of cloud droplets (used in KK2000, B1994, TC1980, LD2004, VarTimeScaleAcnv and SB2006 rain formation schemes)
        "prescribed_Nd" => Float64(1e8),

        # Set to true if you want to assume an open system for aerosol activation with an aerosol concentration budget equal to prescribed_Nd
        "open_system_activation" => false,

        # Set to true if you want to apply ARG aerosol activation locally otherwise activation occurs at the cloud base
        "local_activation" => false,

        # Set to true if you want to generate some basic plots at the end of the simulation
        "plotting_flag" => true,

        # Set to true if you want to switch on autoconversion and accretion in the 1-moment scheme, or collisional coalescence in Cloudy
        "precip_sources" => true,

        # Set to true if you want to switch on evaporation, deposition, sublimation and melting in the 1-moment scheme; or condensation/evaporation in Cloudy
        "precip_sinks" => true,

        # Set to true if you want to apply flux correction for advecting q_tot.
        # (By default flux correction is not applied to q_tot but is applied to all other microphysics tracers)
        "qtot_flux_correction" => false,

        # Bottom of the computational domain [m]
        "z_min" => Float64(0.0),

        # Top of the computational domain [m]
        "z_max" => Float64(2000),

        # Number of computational elements
        "n_elem" => 128,

        # Simulation time step [s]
        "dt" => Float64(1),

        # Output time step [s]
        "dt_output" => Float64(30),

        # Time at the beginning of the simulation [s]
        "t_ini" => Float64(0),

        # Time at the end of the simulation [s]
        "t_end" => Float64(3600),

        # Maximum prescribed updraft momentum flux [m/s * kg/m3]
        "w1" => Float64(2),

        # Oscillation time of the prescribed momentum flux [s]
        "t1" => Float64(600),

        # Pressure at the surface [pa]
        "p0" => Float64(100000),

        # aerosol distribution mean radius for aerosol activation calculations in 2M schemes [m]
        "r_dry" => Float64(0.04 * 1e-6),

        # aerosol distribution standard deviation for aerosol activation calucaulations in 2M schemes
        "std_dry" => Float64(1.4),

        # hygroscopicity of aerosols for aerosol activation calucaulations in 2M schemes
        "kappa" => Float64(0.9),

        # Initial condition z0 [m]
        "z_0" => Float64(0),

        # Initial condition z1 [m]
        "z_1" => Float64(740),
        
        # Initial condition z2 [m]
        "z_2" => Float64(3260),

        # Initial condition rv0 [kg/kg]
        "rv_0" => Float64(0.015),

        # Initial condition rv1 [kg/kg]
        "rv_1" => Float64(0.0138),

        # Initial condition rv2 [kg/kg]
        "rv_2" => Float64(0.0024),

        # Initial condition theta0 [K]
        "tht_0" => Float64(297.9),

        # Initial condition theta1 [K]
        "tht_1" => Float64(297.9),

        # Initial condition theta2 [K]
        "tht_2" => Float64(312.66),

        # Characteristics of particle flux being introduced into P3 domain top (if ice_start = false) or of the initial ice signal (if ice_start = true)
        "p3_boundary_condition" => (;
            ice_start = false,
            _magnitude = Float64(0.5),
            _q_flux = Float64(0.65e-4),
            _N_flux = Float64(40000),
            _F_rim = Float64(0.2),
            _F_liq = Float64(0.2),
            _ρ_r_init = Float64(900),
        )
    )
end