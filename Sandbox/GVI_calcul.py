import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')  # Utilisez le backend non interactif Agg
import matplotlib
matplotlib.use('TkAgg')  # Utilisez TkAgg backend
import matplotlib.pyplot as plt


def gait_parameters_calculation(data, pied, seuil, nbcycle):
    if pied == 1:
        fv_pied1 = data['Force'][0][2]
        fv_pied2 = data['Force'][1][2]
        opp_cal = data['Trajectories'][0]
        study_cal = data['Trajectories'][1]
    else:
        fv_pied1 = data['Force'][1][2]
        fv_pied2 = data['Force'][0][2]
        opp_cal = data['Trajectories'][1]
        study_cal = data['Trajectories'][0]

    fv_pied1 -= np.min(fv_pied1)
    fv_pied2 -= np.min(fv_pied2)

    fs_force = data['Fs'][1]
    fs_mks = data['Fs'][0]
    tmks = np.arange(0, (len(opp_cal[0])) / fs_mks, 1 / fs_mks)
    tforce = np.arange(0, len(fv_pied1) / fs_force, 1 / fs_force)

    idx_all = np.where(np.diff(fv_pied1 > seuil) == 1)[0]
    idx_deb = idx_all[::2]
    idx_fin = idx_all[1::2]
    if abs(idx_fin[0] - idx_deb[0]) < 1200:
        idx_deb = idx_all[1::2]
        idx_fin = idx_all[::2]
    idx_fin = np.delete(idx_fin, np.where(idx_fin < idx_deb[0]))

    idx_deb = idx_deb[3:nbcycle + 4]
    idx_fin = idx_fin[3:nbcycle + 3]

    StrideTime = np.diff(idx_deb) / fs_force
    idx_TpfToTframe = fs_mks / fs_force
    StrideLength = abs(study_cal[0][(idx_fin[:]*idx_TpfToTframe).astype(int)]-study_cal[0][(idx_deb[0:-1]*idx_TpfToTframe).astype(int)])+abs(study_cal[0][(idx_deb[1:] * idx_TpfToTframe).astype(int)] - study_cal[0][(idx_fin[:] * idx_TpfToTframe).astype(int)])
    StrideLength = StrideLength / 1000 # passage de mm à m
    StanceTime = (idx_fin - idx_deb[:-1]) / fs_force
    SwingTime = StrideTime - StanceTime

    SingleSupportTime = np.zeros(nbcycle)
    StepTime = np.zeros(nbcycle)
    StepLength = np.zeros(nbcycle)

    for ii in range(nbcycle):
        fin_contact_pied_opp = np.where(fv_pied2[idx_deb[ii]:idx_fin[ii]] < (seuil / 2))[0][0] + idx_deb[ii]
        debut_contact_pied_opp = np.where(fv_pied2[0:idx_fin[ii]] < (seuil / 2))[0][-1]
        SingleSupportTime[ii] = (debut_contact_pied_opp - fin_contact_pied_opp) / fs_force
        StepTime[ii] = (debut_contact_pied_opp - idx_deb[ii]) / fs_force
        idx_deb_contact_opp_tmks = round(debut_contact_pied_opp * (fs_mks / fs_force))
        StepLength[ii] = np.abs(opp_cal[0, idx_deb_contact_opp_tmks] - study_cal[0, idx_deb_contact_opp_tmks])

    DoubleSupportTime = StanceTime - SingleSupportTime
    StepLength=StepLength/1000 # passage de mm à m
    Velocity = StrideLength / StrideTime

    GaitParametres = np.vstack((
                               StepLength, StrideLength, StepTime, StrideTime, SwingTime, StanceTime, SingleSupportTime,
                               DoubleSupportTime, Velocity)).T

    return GaitParametres


def estimation_mean_std_abs_diff_per(data):
    leg_names = ['right', 'left']
    result = {}
    for leg in leg_names:
        value = data[leg + '_leg']
        moy= np.mean(value, axis=0)
        percentage = (value / moy) * 100
        abs_diff = np.abs(np.diff(percentage,axis=0))
        mean_abs_diff = np.mean(abs_diff, axis=0)
        std_abs_diff = np.std(abs_diff, axis=0)
        result[leg + '_leg'] = {'mean_abs_diff': mean_abs_diff, 'std_abs_diff': std_abs_diff}
    return result


def compute_egvi(data):
    coeff = np.array(
        [-0.797000842686978, -0.698607053374179, -0.930141298777562, -0.900995265111056, -0.898723124870476,
         -0.919179457467057, -0.902164341913704, -0.804781571728659, -0.889550879848708, -0.726249204979732,
         -0.727755402969509, -0.816899569759537, -0.836330324036393, -0.882013306612553, -0.852288965613789,
         -0.860466293197901, -0.68684152584714, -0.898833324436834])
    leg_names = ['right', 'left']
    keGVI = {}
    for leg in leg_names:
        value = np.concatenate((data[leg + '_leg']['mean_abs_diff'], data[leg + '_leg']['std_abs_diff']))
        ponderate_value = np.dot(coeff, value)
        ref = -39.29
        d_sain_study = np.abs(ponderate_value - ref)
        score = np.log(d_sain_study)
        z_score = (score - 100.3) / 7.6
        keGVI[leg + '_leg'] = 100 - 10 * z_score
    return keGVI



# Charger les données depuis le fichier MAT
data = scipy.io.loadmat('C:\\Users\\felie\\Documents\\PostDoc_eWalking\\Data_old\\Test_marche')
trajectories = data['Test_marche_modele_appli'][0]['Trajectories'][0][0][0][0][0][0]
trajectories_label=trajectories[1][0]
trajectories_data=trajectories[2]

index_RCAL = np.where(trajectories_label == 'RCAL')[0]
trajectories_data_RCAL=trajectories_data[index_RCAL[0]]

index_LCAL = np.where(trajectories_label == 'LCAL')[0]
trajectories_data_LCAL=trajectories_data[index_LCAL[0]]

forces = data['Test_marche_modele_appli'][0]['Force']
PF2=forces[0][0][1][5]
PF1=forces[0][0][0][5]

data2 = {}

# Initialisez la clé 'Force' avec une liste vide
data2['Force'] = []
data2['Trajectories']=[]
data2['Fs']=[]

# Ajouter différentes données à la liste 'Force'
data2['Force'].append(PF1)
data2['Force'].append(PF2)
data2['Trajectories'].append(trajectories_data_RCAL)
data2['Trajectories'].append(trajectories_data_LCAL)
data2['Fs'].append(data['Test_marche_modele_appli'][0]['FrameRate'][0][0][0])
data2['Fs'].append(data['Test_marche_modele_appli'][0]['Force'][0][0][0][4][0][0])


#x=list(range(1, len(trajectories_data_RCAL[2]) + 1))
#plt.plot(x, trajectories_data_RCAL[2])

# Calculer les paramètres de la démarche pour la jambe droite et gauche
gait_parameters_left = gait_parameters_calculation(data2, 1, 40, 20)
gait_parameters_right = gait_parameters_calculation(data2, 2, 40, 20)


# Estimer la moyenne et l'écart-type des différences absolues pour chaque paramètre de la démarche
result = estimation_mean_std_abs_diff_per({'right_leg': gait_parameters_right, 'left_leg': gait_parameters_left})

# Calculer l'indice de variabilité de la démarche (GVI)
keGVI = compute_egvi(result)

print(keGVI)