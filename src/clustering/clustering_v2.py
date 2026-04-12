import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler


class APCluster:
    def __init__(self, features, device_names):
        self.features = features
        self.device_names = device_names

    def Affinity_Propagation(self):
        """
        features     : shape (N, F)
        device_names : list of length N

        return:
        {
            "level 1": [device weakest → strongest],
            "level 2": [...],
            ...
        }
        """

        assert len(self.features) == len(self.device_names), "Mismatch input size"

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.features)

        # Affinity Propagation clustering
        AP = AffinityPropagation(
            random_state=42
        )
        labels = AP.fit_predict(X_scaled)


        # Power score
        power_score = [gflops for gflops,_ in X_scaled]

        # Group devices by cluster-
        cluster_map = {}
        for name, label, score in zip(self.device_names, labels, power_score):
            cluster_map.setdefault(label, []).append((name, score))

        # 5. Sort clusters by sum gflops
        cluster_order = sorted(
            cluster_map.keys(),
            key=lambda c: np.sum([s for _, s in cluster_map[c]])
        )

        # Build result dict
        result = {}
        for i, c in enumerate(cluster_order, start=1):
            # sort devices inside cluster
            devices_sorted = sorted(
                cluster_map[c],
                key=lambda x: x[1]
            )
            result[f"level {i}"] = [d[0] for d in devices_sorted]

        return result

    def run(self):
        return self.Affinity_Propagation()


class Clustering:
    # input :

    # lst_devices[0, [0, 'c74b1a5f-5bc6-4946-992b-8f97f5826469'], [0, 'f3b612ea-5a9d-485f-be90-b199e0e1d0bd']]

    # data_clients
    # {'c74b1a5f-5bc6-4946-992b-8f97f5826469': {
    #     'device': {'Total Ram': 16, 'Total Storage': 840, 'Internet': 4000, 'Core': 64}, 'stage': 1},
    #  'f3b612ea-5a9d-485f-be90-b199e0e1d0bd': {
    #      'device': {'Total Ram': 48, 'Total Storage': 420, 'Internet': 2000, 'Core': 32}, 'stage': 2}}

    # return
    # dict with key : uuid and value : cluster_id
    def __init__(self , lst_devices , data_clients):
        self.lst_devices = lst_devices
        self.data_clients = data_clients
        self.dict_res = {}   # dict store each str uuid correspond to cluster

    def extract_device_info(self , dict_devices ) :    # return list info
        lst_data = []
        for values in dict_devices.values() :
            lst_data.append(values)
        return  lst_data

    def run(self):

            id_names = []
            features = []
            for client_id in self.lst_devices[1]:
                if client_id != 0  :
                    id_names.append(client_id)
                    features.append(self.extract_device_info(self.data_clients[client_id]['device']))


            cluster = APCluster(
                features=features,
                device_names=id_names,
            )

            res = cluster.run()
            # RES OF CLUSTERING
            # {'level 1': ['6944d2d1-f8c2-43a6-a023-d5fd3e5727c0'], 'level 2': ['1b084d14-c818-49cf-90a9-3157803fd32d']}

        #     for level in res.keys():
        #         for client_id in res[level]:
        #             self.dict_res[client_id] = int(level[-1])
        #
        # return self.dict_res



#  Example feature matrix (N=5 devices, F=3 features)
# [ GFLOPs, Bandwidth ]
# features_edge = np.array([
#     [1.5,93.5],
#     [3,96.3],
#     [5,94.2],
#     [7,89.8],
#     [9.5,96.9],
# ])
#
# # Device names (must match number of rows)
# device_names = [
#     "Device_A",
#     "Device_B",
#     "Device_C",
#     "Device_D",
#     "Device_E"
# ]