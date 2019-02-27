import datetime

def get_datetime_str():
    return datetime.datetime.today().strftime('%m.%d.%Y-%H:%M:%S')

class Sample(object):

    # simply handles attributes stored in csv files

    def __init__(self,csv_line):
        number,bmi,gender,is_training,name = csv_line.split(',')
        self.number = int(number)
        self.bmi = float(bmi)
        self.gender = gender
        self.is_training = bool(is_training)
        self.file_name = file_name

    def get_in_csv_format(self):
        return ','.join([str(self.number),
                         str(self.bmi),
                         self.gender,
                         str(int(self.is_training)),
                         self.file_name])



def read_samples_from_file(file_path):
    with open(file_path, 'r') as f:
        file_lines = f.read().splitlines()[1:]
    return [Sample(l) for l in file_lines]

def write_samples_to_file(samples, file_path):
    with open(file_path, 'w') as f:
        f.write(',bmi,gender,is_training,name\n')
    for s in samples:
        with open(file_path, 'a') as f:
            f.write('%s\n' % s.get_in_csv_format())
