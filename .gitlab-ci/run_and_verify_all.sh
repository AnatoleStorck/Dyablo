set +e
set -x
set -o pipefail

err_count=0
run_count=0
out_list=()
failure_reason=()
reason_success=0
reason_run_failure=1
reason_verification_failure=2

run_and_verify(){
    ini_file=$1
    verification_script=$2
    out=$3
    mkdir $out
    run_stdout_filename=$out/run.txt
    verification_stdout_filename=$out/verification.txt
    echo "Testing : Execution logs in artifact $run_stdout_filename"
    .gitlab-ci/run_testcase_mpi.sh $ini_file 1 &> $run_stdout_filename
    if [ $? -eq 0 ]
    then
	    echo "run ${ini_file} success"
        echo "Validate : $verification_script"
        cd build/dyablo/bin
        python3 $verification_script &> ../../../$verification_stdout_filename
        if [ $? -eq 0 ]
        then
            success_list+=(${reason_success})
        else
            echo "verification ${verification_script} fail : see ${verification_stdout_filename}"
            err_count=$((err_count+1))
            success_list+=(${reason_verification_failure})
        fi
        cd ../../..
    else
	    echo "run ${ini_file} fail : see ${run_stdout_filename}"
	    err_count=$((err_count+1))
        success_list+=(${reason_run_failure})
    fi 
    out_list+=($out)
    ini_list+=($ini_file)
    run_count=$((run_count+1))
}

run_and_verify test_sod_2D.ini "validate_sod.py test_sod_2D_main.xmf 1e-2 ../../../sod_2D/sod.png" sod_2D 


cd build/dyablo/bin
python3 ../../../settings/cosmo/zeldovitch_generate_grafic.py
cd ../../..

run_and_verify test_zeldovitch_grafic.ini "../../../settings/cosmo/validate_zeldovitch.py zeldovitch_main.xmf 0.2 ../../../zeldovitch_grafic/zeldovitch_grafic.png" zeldovitch_grafic 

run_and_verify test_zeldovitch_dyablo.ini "../../../settings/cosmo/validate_zeldovitch.py zeldovitch_main.xmf 0.2 ../../../zeldovitch_dyablo/zeldovitch_dyablo.png" zeldovitch_dyablo 
run_and_verify test_zeldovitch_particles_dyablo.ini "../../../settings/cosmo/validate_zeldovitch.py zeldovitch_main.xmf 0.2 ../../../zeldovitch_particles_dyablo/zeldovitch_particles_dyablo.png" zeldovitch_particles_dyablo 
run_and_verify beam.ini "../../../settings/cosmo/validate_beam.py beam_main.xmf 1e-2 ../../../beam/beam.png" beam 
run_and_verify test_heat_conduction.ini "validate_heat_conduction.py . 0.02 ../../../heat_conduction/heat_conduction.png" heat_conduction

run_and_verify test_star_formation.ini "validate_star_formation.py star_formation_main.xmf 1e-2 ../../../star_formation/star_formation.png" star_formation

# Create cooling table
cd build/dyablo/bin
python3 ../../../settings/cooling/create_analytical_cooling_table.py
cd ../../..

run_and_verify test_cooling.ini "validate_cooling.py cooling_main.xmf 0.05 ../../../cooling/cooling.png" cooling
run_and_verify test_star_feedback.ini "validate_star_feedback.py star_feedback_main.xmf ../../../star_feedback/star_feedback.png" star_feedback

echo "${err_count}/${run_count} runs failed"


echo "<testsuites tests=\"${run_count}\" failures=\"${err_count}\" disabled=\"0\" errors=\"${err_count}\" name=\"AllTests\">" > report.xml
echo "  <testsuite name=\"DyabloRun\" tests=\"${run_count}\" failures=\"${err_count}\" disabled=\"0\" skipped=\"0\" errors=\"${err_count}\" >" >> report.xml
for ((i = 0 ; i < ${run_count} ; i++ ));
do
echo "    <testcase name=\"${ini_list[$i]}\">" >> report.xml
echo "      <system-out>" >> report.xml
echo "[[ATTACHMENT|${out_list[$i]}/run.txt]] :  " >> report.xml
echo "$(tail -n 25 ${out_list[$i]}/run.txt)  " >> report.xml
echo "[[ATTACHMENT|${out_list[$i]}/verification.txt]] :" >> report.xml
echo "$(tail -n 25 ${out_list[$i]}/verification.txt)  " >> report.xml
echo "      </system-out>" >> report.xml
if [ ${success_list[$i]} -ne 0 ]
then
echo "        <failure />" >> report.xml
fi
echo "    </testcase>" >> report.xml
done

echo "  </testsuite>" >> report.xml
echo "</testsuites>" >> report.xml

exit $err_count

