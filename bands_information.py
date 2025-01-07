import auxil
import numpy as np
import scipy.io as scio
def opbs(image_data, sel_band_count, removed_bands=None):
    if image_data is None:
        return None
    bands = image_data.shape[1]
    band_idx_map = np.arange(bands)
    if not (removed_bands is None):
        image_data = np.delete(image_data, removed_bands, 1)
        bands = bands - len(removed_bands)
        band_idx_map = np.delete(band_idx_map, removed_bands)
    # Compute covariance and variance for each band
    # TODO: data normalization to all band
    data_mean = np.mean(image_data, axis=0)
    image_data = image_data - data_mean
    data_var = np.var(image_data, axis=0)
    h = data_var * image_data.shape[0]
    op_y = image_data.transpose()

    sel_bands = np.array([np.argmax(data_var)])
    last_sel_band = sel_bands[0]
    current_selected_count = 1
    sum_info = h[last_sel_band]
    while current_selected_count < sel_band_count:
        for t in range(bands):
            if not (t in sel_bands):
                op_y[t] = op_y[t] - np.dot(op_y[last_sel_band], op_y[t]) / h[last_sel_band] * op_y[last_sel_band]
        max_h = 0
        new_sel_band = -1
        for t in range(bands):
            if not (t in sel_bands):
                h[t] = np.dot(op_y[t], op_y[t])
                if h[t] > max_h:
                    max_h = h[t]
                    new_sel_band = t
        sel_bands = np.append(sel_bands, new_sel_band)
        last_sel_band = new_sel_band
        sum_info += max_h
        estimate_percent = sum_info / (sum_info + (bands - sel_bands.shape[0]) * max_h)
        print(estimate_percent)
        current_selected_count += 1
    print(band_idx_map[sel_bands] + 1)
    print(np.sort(band_idx_map[sel_bands] + 1))
    return sel_bands


def main():
    use_mat_data = True
    if use_mat_data:
        data, mask = auxil.loadData('HG', num_components=False)
        image_data = data

        cols = image_data.shape[1]
        rows = image_data.shape[0]
        bands = image_data.shape[2]
        image_data = image_data.reshape(cols * rows, bands)
    opbs(image_data, 160)
if __name__ == "__main__":
    main()