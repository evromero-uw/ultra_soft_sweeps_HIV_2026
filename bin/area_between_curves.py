import numpy as np
import pandas as pd
###############################################################################
# Helper functions to handle sub calculations for area between curves.
###############################################################################
def areas_neg(z, dx, dx_intersect):
    """
    This function calculates the area between two curves when they intersect. 
    It takes in the z values (the difference between the two curves at each point),
    the distance between the x values (dx), and the distance from the left x value
    to the intersection point (dx_intersect).
    
    :param z: A numpy array of the difference between the two curves at each x point.
    :param dx: A numpy array of the distance between the x values.
    :param dx_intersect: A numpy array of the distance from the left x value to the intersection point.
    """
    #if the day 0 curve is above the later curve, then the area is the area under the day 0 curve plus 
    if z[1] >= 0:
        return 0.5 * dx_intersect * abs(z[:-1]) - 0.5 * (dx - dx_intersect) * abs(z[1:])
    elif z[1] < 0:
        return -0.5 * dx_intersect * abs(z[:-1]) + 0.5 * (dx - dx_intersect) * abs(z[1:])

def areas_pos(z, dx):
    """
    This function calculates the area between two curves when they do not 
    intersect. It uses the formula for the area of a trapezoid to calculate
    the area between the two curves.
    
    :param z: A numpy array of the difference between the two curves at each
              x point.
    :param dx: A numpy array of the distance between the x values.
    """
    #z[:-1] and z[1:] are the bases of the trapezoid, and dx is the height. 
    area_result = abs(z[:-1] + z[1:]) * 0.5 * dx # signs of both z are same

    #If the day 0 curve is above the later curve, then the area is negative,
    #otherwise it is positive.
    if z[1] >= 0:
        return -1 * area_result
    elif z[1] < 0:
        return area_result
    
###############################################################################
# Main function to calculate area between curves.
###############################################################################
    
def area_between_curves(x, y1, y2):
    """This function calculates the area between two curves given their x 
    and y values. In this function, y1 is the day 0 curve (which is the baseline)
    and y2 is the later curve. Positive area values indicate that the day 0
    curve below the later curve.    
    """
    # Z holds the height of the area between the two curves at each x point.
    # The sign of z indicates which curve is above the other.
    z = y1-y2

    # dx holds the distance between the x values
    dx = x[1:] - x[:-1]

    # cross_test holds the sign of the product of the z values at adjacent
    # x points. If the product is negative, it means that the curves
    # intersect between those two x points.
    cross_test = np.sign(z[:-1] * z[1:])

    # dx_intersect holds the distance from the left x value to the 
    # intersection point
    dx_intersect = - dx / (z[1:] - z[:-1]) * z[:-1]

    #This list will hold the area calculations for each segment between x points.
    area_calculations = []

    #Loop through the test results
    for j in range(len(cross_test)):
        #If the lines cross then we need to calculate the area of two triangles
        if cross_test[j] < 0:
            area_neg = areas_neg(z[j:j+2], dx[j], dx_intersect[j])
            area_calculations.append(area_neg)
        #Otherwise, we can calculate the area of a trapezoid
        elif cross_test[j] >= 0:
            area_pos = areas_pos(z[j:j+2], dx[j])
            area_calculations.append(area_pos)

    return np.sum(area_calculations)


###############################################################################
# Convenience wrapper for binned Pre/Post iHH curves.
###############################################################################

def area_between_binned_curves(rep_data, fragment_length=2500):
    """Merge one replicate's Pre/Post binned-iHH curves on their bins and return
    the area between them, normalized by fragment_length.

    :param rep_data: A DataFrame for a single replicate with columns time_label,
                     bin_start, bin_end, binned_average and adj_iHH_std.
    :param fragment_length: Length used to normalize the area so it is comparable
                            across parameter combinations.
    """
    pre_curve = rep_data[rep_data['time_label'] == 'Pre'].sort_values(by='bin_start')
    post_curve = rep_data[rep_data['time_label'] == 'Post'].sort_values(by='bin_start')

    # Merge on the bin coordinates so the two curves are aligned for the area calc.
    merged = pd.merge(pre_curve, post_curve, on=['bin_start', 'bin_end'],
                      suffixes=('_d0', '_post'))
    merged = merged.dropna(subset=['binned_average_d0', 'binned_average_post',
                                   'adj_iHH_std_d0', 'adj_iHH_std_post'])

    area = area_between_curves(merged['bin_start'].values,
                               merged['binned_average_d0'].values,
                               merged['binned_average_post'].values)
    return area / fragment_length