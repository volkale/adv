DRUG_CLASSES = ['atypical', 'ssri', 'ssnri', 'tca']


def get_drug_class(drug):
    dose_indep_drug = drug.split(' ')[0].lower()
    if dose_indep_drug in ['sertraline', 'paroxetine', 'citalopram', 'escitalopram', 'fluoxetine', 'fluvoxamine']:
        return 'ssri'
    elif dose_indep_drug in ['venlafaxine', 'desvenlafaxine', 'duloxetine', 'milnacipran', 'levomilnacipran']:
        return 'ssnri'
    elif dose_indep_drug in ['amitriptyline', 'clomipramine']:
        return 'tca'
    elif dose_indep_drug in [
        'trazodone', 'nefazodone', 'vortioxetine', 'agomelatine', 'bupropion', 'mirtazapine', 'reboxetine', 'vilazodone'
    ]:
        return 'atypical'
    elif dose_indep_drug == 'placebo':
        return 'placebo'
    else:
        return None
