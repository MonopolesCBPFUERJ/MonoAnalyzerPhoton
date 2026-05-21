source /cvmfs/sft.cern.ch/lcg/views/LCG_106a/x86_64-el9-gcc11-opt/setup.sh
python3 -c "import mplhep, uproot, numpy, matplotlib; print('OK')"

python3 plot_limits.py --preset rateParam_Combined_all_systs --misc all_systs --strategy Combined --no-annotate-intersections
#python3 plot_limits.py --preset rateParam_PFMET_all_systs --misc all_systs --strategy PFMET --no-annotate-intersections
#python3 plot_limits_updated.py --preset Photon_all_systs --misc all_systs --strategy Photon --no-annotate-intersections
