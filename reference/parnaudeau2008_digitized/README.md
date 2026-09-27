# Digitized from Parnaudeau et al. 2008, Figs. 9-10

Source: P. Parnaudeau, J. Carlier, D. Heitz, E. Lamballais, "Experimental and numerical studies of the
flow over a circular cylinder at Reynolds number 3900," Phys. Fluids 20, 085101 (2008). Open access:
https://www.irisa.fr/fluminance/team/Carlier/publications/ParnaudeauCarlierHeitzLamballaisPOF.pdf

No machine-readable dataset is published (the paper's own reference 31: "available by contacting the
authors"). These CSVs were produced by pixel-level digitization of the vector-rendered PDF page
(400 dpi, `pdftoppm`), calibrated against the axes' own tick-mark pixel positions (not eyeballed): for
each x pixel column inside the plot frame, the median, 15th and 85th percentile of all dark-pixel data
values in that column, converted to data coordinates. Fig. 9's plot carries eight series (present LES,
present PIV, present HWA, Kravchenko & Moin's B-spline LES, Ong & Wallace, Lourenco & Shih, Norberg at
two Re, Ma et al.'s DNS); the median trace is therefore a consensus of all of them, not exclusively the
PIV series, though the PIV points dominate by sheer density in x/D < 3 where the paper says their PIV
field of view lives (Table I). Sanity-checked against the paper's own Table II (text, not digitized):
digitized U_min -0.285 at x/D 1.51 (Table II PIV: -0.34, no location given); digitized <u'u'> peak 0.104
at x/D 1.42 (Table II PIV: L_<u'u'> = 0.87 D from the base, i.e. x/D = 1.37 -- 0.05 D from the digitized
peak). Uncertainty: pixel resolution ~0.01 D in x, ~0.003 in the plotted quantity, plus whatever spread
the p15-p85 columns show (the disagreement between the plot's eight series at that x). Past x/D ~ 4.5
(beyond the PIV field of view) the only remaining series are a few sparse, hollow-ring markers (HWA
circles); a ring crossing one pixel column leaves two disconnected dark clusters whose median jumps
between them column to column, producing visible jitter in that sparse far-wake region -- real, and
confined to where the comparison matters least (the near-wake / recirculation region, x/D < 3, is
dense and smooth). Past x/D ~ 4.5
(beyond the PIV field of view) the only remaining series are a few sparse, hollow-ring markers (HWA
circles); a ring crossing one pixel column leaves two disconnected dark clusters whose median jumps
between them column to column, producing visible jitter in that sparse far-wake region -- real, and
confined to where the comparison matters least (the near-wake / recirculation region, x/D < 3, is
dense and smooth).

Files: `fig9_centerline_u.csv` (mean streamwise velocity, wake centreline, x/D from the cylinder
centre), `fig10_centerline_uu.csv` (its variance). Columns: x_D, {u,uu}_median, _p15, _p85, n_dark_px
(pixels found in that column -- low n means sparse/no data at that x, high n means the dense PIV cluster).
Script: `plot_utility/plot_ucylinder_re3900_vs_piv.py`.
