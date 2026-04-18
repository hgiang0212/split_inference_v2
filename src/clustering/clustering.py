import numpy as np
from sklearn.cluster import AffinityPropagation
import sys


class APCluster:
    def __init__(self, features, device_names, alpha=1, nums_cloud=1):
        self.features = features
        self.device_names = device_names
        self.alpha = alpha
        self.nums_cloud = nums_cloud

    def Affinity_Propagation(self):
        """
        features     : shape (N, F)
        device_names : list of length N

        return:
        {
            res = { 0 : {  "name_devices" : [device A , device B , ...],
                           "nums_cloud" : 2,
                           "mean_score" : (0.5,0.3) }
                    1 : ...}
        }
        """

        assert len(self.features) == len(self.device_names), "Mismatch input size"

        # Scale features
        min_values = self.features.min(axis=0)
        max_values = self.features.max(axis=0)
        eps = 1e-8
        features_scaled = (self.features - min_values) / (max_values - min_values + eps)

        # Affinity Propagation clustering
        AP = AffinityPropagation(preference=None, random_state=42)
        labels = AP.fit_predict(features_scaled)
        labels = labels.astype(int).tolist()
        nums_cluster = len(AP.cluster_centers_indices_)
        if (self.nums_cloud < nums_cluster):
            print("Warning! The number of cloud < the number of cluster")
            sys.exit()

        # Group devices by cluster
        cluster_map = {}
        for name, label, score in zip(self.device_names, labels, features_scaled):
            group = cluster_map.setdefault(label, {"name_devices": [], "score_devices": []})
            group["name_devices"].append(name)
            group["score_devices"].append(score)

        # Calculating strength of cluster
        strength_cl = {}
        mean_gflops_cl = {}
        mean_bandwidth_cl = {}
        cloud_2_cl = {}
        for label, data in cluster_map.items():
            scores = data["score_devices"]
            sum_gflops_cl, sum_bw_cl = np.sum(scores, axis=0)
            strength_cl[label] = self.alpha * sum_gflops_cl + (1 - self.alpha) * sum_bw_cl
            mean_gflops_cl[label], mean_bandwidth_cl[label] = np.mean(scores, axis=0)
            cloud_2_cl[label] = 1

        # Match cloud to cluster
        r = self.nums_cloud - nums_cluster
        while r > 0:
            p = 0
            i = 0
            for label, strength in strength_cl.items():
                pressure = strength / cloud_2_cl[label]
                if pressure > p:
                    p = pressure
                    i = label
            cloud_2_cl[i] += 1
            r -= 1

            # Build result dict
        result = {}
        for label, data in cluster_map.items():
            result[label] = {}
            result[label] = {
                "name_devices": data["name_devices"],
                "nums_cloud": cloud_2_cl[label],
                "mean_score": (mean_gflops_cl[label], mean_bandwidth_cl[label])
            }
        return result

    def run(self):
        return self.Affinity_Propagation()


class Clustering:
    # input :

    # lst_devices[0, [0, 'c74b1a5f-5bc6-4946-992b-8f97f5826469'], [0, 'f3b612ea-5a9d-485f-be90-b199e0e1d0bd']]

    # data_clients
    # {'c74b1a5f-5bc6-4946-992b-8f97f5826469': {
    #     'device': {"GFLOPs": 1.5 ,"Internet": 100}, 'stage': 1},
    #  'f3b612ea-5a9d-485f-be90-b199e0e1d0bd': {
    #      'device': {"GFLOPs": 6 ,"Internet": 300}, 'stage': 2}}

    # return
    # dict with key : uuid and value : cluster_id
    def __init__(self, lst_devices, data_clients):
        self.lst_devices = lst_devices
        self.data_clients = data_clients
        self.dict_res = {}  # dict store each str uuid correspond to cluster

    def extract_device_info(self, dict_devices):  # return list info
        lst_data = []
        for values in dict_devices.values():
            lst_data.append(values)
        return lst_data

    def run(self):

        id_names = []
        features = []
        for client_id in self.lst_devices[1]:
            if client_id != 0:
                id_names.append(client_id)
                features.append(self.extract_device_info(self.data_clients[client_id]['device']))
        nums_cloud = len(self.lst_devices[2]) - 1
        features = np.array(features)
        cluster = APCluster(
            features=features,
            device_names=id_names,
            nums_cloud=nums_cloud,
        )

        res = cluster.run()

        cloud_idx = 1
        for cluster in res.keys():
            for edge_device in res[cluster]["name_devices"]:
                self.dict_res[edge_device] = cluster
            for j in range(res[cluster]["nums_cloud"]):
                self.dict_res[self.lst_devices[2][cloud_idx]] = cluster
            cloud_idx += 1
        print(self.dict_res)
        return self.dict_res

#  Example feature matrix (N=5 devices, F=3 features)
# [ GFLOPs, Bandwidth ]
# features_edge = np.array([
#     [1.5 , 100],
#     [3 , 200],
#     [5 , 300],
#     [7 , 400],
#     [9 , 600],
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