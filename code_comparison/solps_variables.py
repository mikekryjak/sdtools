
vars = {
    
    ##########################################################################
    # GEOMETRY
    ##########################################################################
    
    "crx" : dict(
        name = "R coordinates of cell corners",
        units = "m",
        notes = "Order: lower left, lower right, upper left, upper right. Centre missing",
    ),
    "cry" : dict(
        name = "Z coordinates of cell corners",
        units = "m",
        notes = "Order: lower left, lower right, upper left, upper right. Centre missing",
    ),
    
    ##########################################################################
    # EIRENE
    ##########################################################################
    
    "eirene_mc_papl_sna_bal" : dict(
        name = "Particle source (bulk ions) from atom-plasma coll.",
        units = "A cm-3",
        notes = "",
        origin = "balance.nc"
    ),
    
    "eirene_mc_pmpl_sna_bal" : dict(
        name = "Particle source (bulk ions) from molecule-plasma coll.",
        units = "A cm-3",
        notes = "",
        origin = "balance.nc"
    ),
    
    "eirene_mc_pipl_sna_bal" : dict(
        name = "Particle source (bulk ions) from test ion-plasma coll.",
        units = "A cm-3",
        notes = "",
        origin = "balance.nc"
    ),
    
    "eirene_mc_pppl_sna_bal" : dict(
        name = "Primary particle sources rate (Bulk ions)",
        units = "A cm-3",
        notes = "",
        origin = "balance.nc"
    ),
    
    "wldnek" : dict(
        name = "Heat transferred by neutrals",
        units = "W",
        notes = "Per stratum",
        origin = "fort.44"
    ),
    
    "wldnep" : dict(
        name = "Potential energy released by neutrals",
        units = "W",
        notes = "Per stratum",
        origin = "fort.44"
    ),
    
    "wldna" : dict(
        name = "Flux of atoms impinging on surface",
        units = "A",
        notes = "Per stratum",
        origin = "fort.44"
    ),
    
    "ewlda" : dict(
        name = "Average energy of impinging atoms on surface",
        units = "eV",
        notes = "Per stratum",
        origin = "fort.44"
    ),
    
    "wldnm" : dict(
        name = "Flux of molecules impinging on surface",
        units = "A",
        notes = "Per stratum",
        origin = "fort.44"
    ),
    
    "ewldnm" : dict(
        name = "Average energy of molecules impinging on surface",
        units = "eV",
        notes = "Per stratum",
        origin = "fort.44"
    ),
    
    "wldra" : dict(
        name = "Flux of reflected atoms from surface",
        units = "A",
        notes = "Per stratum",
        origin = "fort.44"
    ),
    
    "wldrm" : dict(
        name = "Flux of reflected molecules from surface",
        units = "A",
        notes = "Per stratum",
        origin = "fort.44"
    ),
    
    "wldpp" : dict(
        name = "Flux of plasma ions impinging on surface",
        units = "A",
        notes = "Per stratum",
        origin = "fort.44"
    ),
    
    "wldpa" : dict(
        name = "Net flux of atoms emitted from surface",
        units = "A",
        notes = "Per stratum",
        origin = "fort.44"
    ),
    
    "wldpm" : dict(
        name = "Net flux of molecules emitted from surface",
        units = "A",
        notes = "Per stratum",
        origin = "fort.44"
    ),
    
    "wldpeb" : dict(
        name = "Power carried by particles emitted from surface",
        units = "W",
        notes = "Per stratum",
        origin = "fort.44"
    ),
    
    "wlarea" : dict(
        name = "Surface area",
        units = "m2",
        notes = "",
        origin = "fort.44"
    ),
    
    ##########################################################################
    # display_tallies
    ##########################################################################
    
    
    "rdneureg" : dict(
        name = "Total radiation from EIRENE neutrals",
        units = "W",
        notes = "",
        origin = "display_tallies"
    ),
    
    "rqradreg" : dict(
        name = "Line radiation",
        units = "W",
        notes = "",
        origin = "display_tallies"
    ),
    
    "rqradreg" : dict(
        name = "Line radiation",
        units = "W",
        notes = "",
        origin = "display_tallies"
    ),
    
    

# eirene_mc_core_sna_bal
# eirene_mc_mapl_smo_bal
# eirene_mc_mmpl_smo_bal
# eirene_mc_mipl_smo_bal
# eirene_mc_cppv_smo_bal
# eirene_mc_paat_sna_bal
# eirene_mc_pmat_sna_bal
# eirene_mc_piat_sna_bal
# eirene_mc_paml_sna_bal
# eirene_mc_pmml_sna_bal
# eirene_mc_piml_sna_bal
# eirene_mc_paio_sna_bal
# eirene_mc_pmio_sna_bal
# eirene_mc_piio_sna_bal
# eirene_mc_pael_sne_bal
# eirene_mc_pmel_sne_bal
# eirene_mc_eael_she_bal
# eirene_mc_emel_she_bal
# eirene_mc_eiel_she_bal
# eirene_mc_epel_she_bal
# eirene_mc_eapl_shi_bal
# eirene_mc_empl_shi_bal
# eirene_mc_eipl_shi_bal
# eirene_mc_eppl_shi_bal
    
}


# crx
# cry
# bb
# hx
# hy
# hz
# vol
# gs
# am
# mp
# ev
# leftix
# leftiy
# rightix
# rightiy
# topix
# topiy
# bottomix
# bottomiy
# jxi
# jxa
# jsep
# species
# b2mndr_eirene
# b2mndr_hz
# za
# fna_pinch
# fna_pll
# fna_drift
# fna_ch
# fna_nanom
# fna_panom
# fna_pschused
# fna_tot
# b2stbr_phys_sna_bal
# b2stbr_bas_sna_bal
# b2stbr_first_flight_sna_bal
# b2stbc_sna_bal
# b2stbm_sna_bal
# ext_sna_bal
# b2stel_sna_ion_bal
# b2stel_sna_rec_bal
# b2stcx_sna_bal
# b2srsm_sna_bal
# b2srdt_sna_bal
# b2srst_sna_bal
# tot_sna_bal
# resco
# fmo_flua
# fmo_cvsa
# fmo_hybr
# fmo_b2nxfv
# fmo_tot
# b2stbr_phys_smo_bal
# b2stbr_bas_smo_bal
# b2stbc_smo_bal
# b2stbm_smo_bal
# ext_smo_bal
# b2stel_smq_ion_bal
# b2stel_smq_rec_bal
# b2stcx_smq_bal
# b2srsm_smo_bal
# b2srdt_smo_bal
# b2srst_smo_bal
# b2sifr_smoch_bal
# b2sifr_smotf_ehxp_bal
# b2sifr_smotf_cthe_bal
# b2sifr_smotf_cthi_bal
# b2sifr_smofrea_bal
# b2sifr_smofria_bal
# b2sifr_smotfea_bal
# b2sifr_smotfia_bal
# b2siav_smovh_bal
# b2siav_smovv_bal
# b2sicf_smo_bal
# b2sian_smo_bal
# b2nxdv_smo_bal
# b2sigp_smogp_bal
# b2sigp_smogpi_bal
# b2sigp_smogpe_bal
# b2sigp_smogpgr_bal
# b2sigp_pstat_bal
# b2sigp_pstati_bal
# b2sigp_pstate_bal
# tot_smo_bal
# resmo
# fhe_32
# fhe_52
# fhe_thermj
# fhe_cond
# fhe_dia
# fhe_ecrb
# fhe_strange
# fhe_pschused
# b2stbr_phys_she_bal
# b2stbr_bas_she_bal
# b2stbr_first_flight_she_bal
# b2stbc_she_bal
# b2stbm_she_bal
# ext_she_bal
# b2stel_she_bal
# b2srsm_she_bal
# b2srdt_she_bal
# b2srst_she_bal
# b2sihs_diae_bal
# b2sihs_divue_bal
# b2sihs_exbe_bal
# b2sihs_joule_bal
# b2npht_shei_bal
# reshe
# fhi_32
# fhi_52
# fhi_cond
# fhi_dia
# fhi_ecrb
# fhi_strange
# fhi_pschused
# fhi_inert
# fhi_vispar
# fhi_visper
# fhi_visq
# fhi_anml
# fhi_kevis
# b2stbr_phys_shi_bal
# b2stbr_bas_shi_bal
# b2stbr_first_flight_shi_bal
# b2stbc_shi_bal
# b2stbm_shi_bal
# ext_shi_bal
# b2stel_she_ion_bal
# b2stel_she_rec_bal
# b2stel_shi_ion_bal
# b2stel_shi_rec_bal
# b2stcx_shi_bal
# b2srsm_shi_bal
# b2srdt_shi_bal
# b2srst_shi_bal
# b2sihs_diaa_bal
# b2sihs_divua_bal
# b2sihs_exba_bal
# b2sihs_visa_bal
# b2sihs_fraa_bal
# reshi
# eirene_mc_papl_sna_bal
# eirene_mc_pmpl_sna_bal
# eirene_mc_pipl_sna_bal
# eirene_mc_pppl_sna_bal
# eirene_mc_core_sna_bal
# eirene_mc_mapl_smo_bal
# eirene_mc_mmpl_smo_bal
# eirene_mc_mipl_smo_bal
# eirene_mc_cppv_smo_bal
# eirene_mc_paat_sna_bal
# eirene_mc_pmat_sna_bal
# eirene_mc_piat_sna_bal
# eirene_mc_paml_sna_bal
# eirene_mc_pmml_sna_bal
# eirene_mc_piml_sna_bal
# eirene_mc_paio_sna_bal
# eirene_mc_pmio_sna_bal
# eirene_mc_piio_sna_bal
# eirene_mc_pael_sne_bal
# eirene_mc_pmel_sne_bal
# eirene_mc_eael_she_bal
# eirene_mc_emel_she_bal
# eirene_mc_eiel_she_bal
# eirene_mc_epel_she_bal
# eirene_mc_eapl_shi_bal
# eirene_mc_empl_shi_bal
# eirene_mc_eipl_shi_bal
# eirene_mc_eppl_shi_bal
# ne
# na
# ua
# po
# te
# ti
# rpt
# kinrgy
# fne
# dab2
# tab2
# rfluxa
# refluxa
# pfluxa
# pefluxa
# dmb2
# tmb2
# rfluxm
# refluxm
# pfluxm
# pefluxm