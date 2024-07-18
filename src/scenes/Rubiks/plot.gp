set terminal pngcairo size 800,600

# Plot Mean Time Taken
set output "mean_time_taken.png"
set title "Mean Time Taken"
set xlabel "Puzzle Type and Method"
set ylabel "Mean Time Taken (seconds)"
set xtics rotate by -45
set grid
plot "experiment_results.dat" using 3:xticlabels(1) with linespoints title 'Mean Time Taken'

# Plot Mean Element Impact
set output "mean_element_impact.png"
set title "Mean Element Impact"
set xlabel "Puzzle Type and Method"
set ylabel "Mean Element Impact"
set xtics rotate by -45
set grid
plot "experiment_results.dat" using 4:xticlabels(1) with linespoints title 'Mean Element Impact'

# Plot Mean Primordial Size
set output "mean_primordial_size.png"
set title "Mean Primordial Size"
set xlabel "Puzzle Type and Method"
set ylabel "Mean Primordial Size"
set xtics rotate by -45
set grid
plot "experiment_results.dat" using 5:xticlabels(1) with linespoints title 'Mean Primordial Size'

# Plot Mean Number of Elements
set output "mean_num_elements.png"
set title "Mean Number of Elements"
set xlabel "Puzzle Type and Method"
set ylabel "Mean Number of Elements"
set xtics rotate by -45
set grid
plot "experiment_results.dat" using 6:xticlabels(1) with linespoints title 'Mean Number of Elements'

# Plot Success Ratio
set output "success_ratio.png"
set title "Success Ratio"
set xlabel "Puzzle Type and Method"
set ylabel "Success Ratio"
set xtics rotate by -45
set grid
plot "experiment_results.dat" using 7:xticlabels(1) with linespoints title 'Success Ratio'

