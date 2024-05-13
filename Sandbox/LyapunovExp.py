import numpy as np
import nolds
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean,pdist, squareform
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import scipy.io
import pyinform


def resize_data(data, min_peak_distance, nbcycle, Fs, NbPtsObj):
    # Trouver les pics dans les données
    peaks, _ = find_peaks(data, distance=min_peak_distance)

    # Extraire les données d'intérêt entre les pics spécifiés
    interest_data = data[peaks[3]:peaks[nbcycle+2]]

    # Créer une échelle de temps pour les données d'intérêt
    time_scale = np.arange(len(interest_data)) / Fs

    # Calculer la fréquence d'échantillonnage appropriée pour les données d'intérêt
    Fs_obj = int(NbPtsObj / time_scale[-1])

    # Créer une échelle de temps avec la nouvelle fréquence d'échantillonnage
    time_scale_resized = np.arange(NbPtsObj) / Fs_obj

    # Interpoler les données pour les redimensionner
    f = interp1d(time_scale, interest_data)
    resized_data = f(time_scale_resized)

    return resized_data



def rosenstein_method(data, delay, dimension):
    # Création de l'espace de phase
    embedded_data = np.array([data[i:(len(data) - dimension * delay + i):delay] for i in range(dimension)])

    # Calcul de la matrice d'état
    state_matrix = np.zeros((embedded_data.shape[1], dimension))
    for i in range(embedded_data.shape[1]):
        state_matrix[i] = embedded_data[:, i]

    # Calcul de la distance entre les paires de points
    pairwise_distances = squareform(pdist(state_matrix))

    # Calcul de la moyenne logarithmique des distances
    sum_log_distances = np.sum(np.log(pairwise_distances + np.eye(pairwise_distances.shape[0])), axis=0)
    mean_log_distances = sum_log_distances / (pairwise_distances.shape[0] - 1)

    # Ajustement linéaire pour calculer l'exposant de Lyapunov
    time = np.arange(0, len(mean_log_distances))
    slope, intercept = np.polyfit(time, mean_log_distances, 1)

    return slope


def estimate_time_delay_ami(data, max_tau):
    ami_values = []
    for tau in range(1,max_tau):
        ami = pyinform.mutualinfo.mutual_info(data[:-tau], data[tau:])
        ami_values.append(ami)
    return np.argmin(ami_values)

def estimate_embedding_dimension(data, delay):
    embed_dim = 5  # Valeur initiale de la dimension d'espace
    pca = PCA(n_components=embed_dim)
    embedding = np.array([data[i:i+delay] for i in range(len(data)-delay)])  # Créer la matrice d'état
    pca.fit(embedding)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    optimal_dim = np.argmax(cumulative_variance_ratio > 0.95) + 1  # Choix du seuil de variance cumulative
    return optimal_dim

def main():
    # Vecteur de coordonnées
    data = scipy.io.loadmat('C:\\Users\\felie\\Documents\\PostDoc_eWalking\\Data_old\\Test_marche')
    trajectories = data['Test_marche_modele_appli'][0]['Trajectories'][0][0][0][0][0][0]

    trajectories_label = trajectories[1][0]
    trajectories_data = trajectories[2]

    index_T10 = np.where(trajectories_label == 'T10')[0]
    trajectories_data_T10 = trajectories_data[index_T10[0]]

    T10Vpos = trajectories_data_T10[2]

    # Fréquence d'échantillonnage (
    Fs = data['Test_marche_modele_appli'][0]['FrameRate'][0][0][0]

    max_tau = 300 # Valeur maximale du retard à tester

    # Estimation du retard temporel optimal
    delay = estimate_time_delay_ami(T10Vpos,max_tau)
    print("Optimal time delay:", delay)

    # Estimation de la dimension d'espace
    optimal_dim = estimate_embedding_dimension(T10Vpos, delay)
    print("Optimal embedding dimension:", optimal_dim)

    slop=rosenstein_method(T10Vpos, delay, optimal_dim)
    print(slop)
    return slop


if __name__ == "__main__":
    slop=main()
