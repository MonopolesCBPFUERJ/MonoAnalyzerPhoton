import os
import glob
import subprocess

def merge_root_files(base_dir, output_dir):
    mass_values = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
    processes = ["SpinHalf_PhotonFusion", "SpinHalf_DrellYan","SpinZero_DrellYan", "SpinZero_PhotonFusion"]
    #mass_values = [1500]
    #processes = ["SpinHalf_DrellYan"]
    
    for mass in mass_values:
        for process in processes:
            target_dir = os.path.join(base_dir, f"Monopole_{process}_M-{mass}_TuneCP5_13TeV_madgraph-pythia8")

            for root, dirs, files in os.walk(target_dir):
                if '0000' in root:
                    output_file = os.path.join(output_dir, f"MM_{process}_M-{mass}_2016APV.root")
                    input_files = glob.glob(os.path.join(root, "output_*"))

                    if not input_files:
                        print(f"No files found in {root} for {process} M-{mass}")
                        continue
                    
                    hadd_command = ["hadd", "-f", output_file] + input_files

                    try:
                        print(f"Merging files for {process} M-{mass} in {root}...")
                        subprocess.run(hadd_command, check=True)
                        print(f"Successfully created: {output_file}")
                    except subprocess.CalledProcessError as e:
                        print(f"Error while merging files in {root}: {e}")

if __name__ == "__main__":
    #base_directory = "/eos/user/t/tmenezes/MM_METcorrected_MC_2018"
    base_directory = "/eos/user/t/tmenezes/MM_METcorrected_MC_2016APV"
    output_directory = "/eos/user/t/tmenezes/Monopole_Ntuples/Central_Production/MET_corrected/2016APV"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    merge_root_files(base_directory, output_directory)

