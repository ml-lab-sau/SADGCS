% Sample data
data = rand(5, 3); % Random 5x3 matrix

% Convert data to a table
T = array2table(data, 'VariableNames', {'Column1', 'Column2', 'Column3'});

% Specify the filename
filename = 'results.xlsx';

% Write the table to an Excel file
writetable(T, filename);
disp(['Results saved to ' filename]);
