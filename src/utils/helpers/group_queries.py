def get_group(input_data):
    grouped_data = input_data.groupby(by='id').agg(lambda x: x.tolist())
    grouped_data['table_file'] = grouped_data['table_file'].apply(lambda x: x[0])
    return grouped_data
