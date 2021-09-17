<!-- # cs4622_ml_project -->
# Machine Learning Project:

Pump It Up | Data mining the water table

Competition hosted by [DrivenData](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/)

## Project Description
Using the data provided by the **Tanzanian Ministry of Water**, develop a solution which can predict which waterpumps are _functional_, _non-functional_ or _functional but needs repair_.

## Dataset Description
- consists details of 59400 waterpumps scattered through Tanzania (used as the train set for developing a solution based on ML)
- Competition challenge is to predict the functionality of 14850 waterpumps

### features in the dataset

    1. amount_tsh - Total static head (amount water available to waterpoint)
    2. date_recorded - The date the row was entered
    3. funder - Who funded the well
    4. gps_height - Altitude of the well
    5. installer - Organization that installed the well
    6. longitude - GPS coordinate
    7. latitude - GPS coordinate
    8. wpt_name - Name of the waterpoint if there is one
    9. num_private -
    10. basin - Geographic water basin
    11. subvillage - Geographic location
    12. region - Geographic location
    13. region_code - Geographic location (coded)
    14. district_code - Geographic location (coded)
    15. lga - Geographic location
    16. ward - Geographic location
    17. population - Population around the well
    18. public_meeting - True/False
    19. recorded_by - Group entering this row of data
    20. scheme_management - Who operates the waterpoint
    21. scheme_name - Who operates the waterpoint
    22. permit - If the waterpoint is permitted
    23. construction_year - Year the waterpoint was constructed
    24. extraction_type - The kind of extraction the waterpoint uses
    25. extraction_type_group - The kind of extraction the waterpoint uses
    26. extraction_type_class - The kind of extraction the waterpoint uses
    27. management - How the waterpoint is managed
    28. management_group - How the waterpoint is managed
    29. payment - What the water costs
    30. payment_type - What the water costs
    31. water_quality - The quality of the water
    32. quality_group - The quality of the water
    33. quantity - The quantity of water
    34. quantity_group - The quantity of water
    35. source - The source of the water
    36. source_type - The source of the water
    37. source_class - The source of the water
    38. waterpoint_type - The kind of waterpoint
    39. waterpoint_type_group - The kind of waterpoint

### labels
    1. functional - the waterpoint is operational and there are no repairs needed
    2. non functional - the waterpoint is not operational
    3. functional needs repair - the waterpoint is operational, but needs repairs

## Feature Engineering

### dataset inspection
- geographical features (categorical)

    | feature | unique value count|
    | --- | --- |
    | district_code | 20 |
    | region | 21 |
    | region_code | 27 |
    | lga | 125 |
    | ward | 2092 |
    | subvllage | 19287 |

    _subvillage_ feature is the most granular categorical representation of waterpump locations

- duplicate features
    | feature | duplicated |
    | --- | --- |
    | payment | payment_type |
    | quantity | quantity_group |

- Not duplicate but similar features
    - extraction_type, extraction_type_class, extraction_type_group
    - source, source_type, source_class
    - waterpoint_type, waterpoint_type_group
    - water_quality, quality_group

- missing values and outliers found in
    - population

        ![population_distribution](assets\population_distribution.png)

        - 21381 rows found with population = 0
        - 7025 rows found with population = 1

        identified above scenarios as outliers because population is most likely to be a distinct feature


    - longitude

        ![longitude_distribution](assets\longitude_distribution.png)

        According to [World Population Review web site](https://worldpopulationreview.com/country-locations/where-is-tanzania), **Tanzania** is located in the world map within the longitude range 29°10' - 40°29'.

        In the dataset, there are 1812 rows with longitude values which are not within the above range.

    - gps_height

        ![gps_height_distribution](assets\gps_height_distribution.png)

        There are 20438 rows with 0 _gps_height_ value which is probably an outlier when lookking at the above distribution.

    - construction_year

        ![construction_year_distribution](assets\construction_year_distribution.png)

        There are 20709 rows with _construction_year_ = 0

    - public_meeting
        ![public_meeting_countplot](assets\public_meeting_countplot.png)

        With respect to the labels, _public_meeting_ feature seems to be decisive for prediction. But there were 3334 rows with `NULL` values.

    - amount_tsh

        41639 rows found with 0 values. Since the percentage of missing values is more than **70%**, decided to remove this feature rather than trying to impute. (which will be less generalized if imputed)

    - funder, installer
        outliers found such as `0`, `-` and the unique value count was huge. (1897 unique values in _funder_ column)

- Less significant features with respect to labels
    - recorded_by   (only 1 unique value)
    - num_private   (large number of unique values)
    - wpt_name      (name of the waterpoint: almost like id)
    - management`*`
    - management_group`*`
    - scheme_name`*`
    - scheme_management`*`
    - permit`*`

    `*` - decided based on countplots with respect to labels

    These features are removed from train and test sets

### Data Imputation Techniques

- iterative imputation based on the granularity of geographic features in the following order.

    1. lga
    2. region_code
    3. district_code
    4. overall

    Outliers of population, gps_height, longitude, construction_year and public_meeting are tranformed into `NaN` beforehand.

    For population, gps_height and longitude features NaN values are replaced based on the mean values in the geographic location the waterpump belonged.

    As an example, the dataset is first grouped by _lga_ and replace the NaN values by mean within the specific _lga_. If there's still NaN values remain in the dataset, replacement is done using the groups of _region_code_ feature. Then grouped by _district_code_ and finally if there's still NaN values left in the data, the overall mean is used to replace them.

    NaN values of _construction_year_ is replaced based on the **median** year within geographic groupings.

    NaN values of _public_meeting_ is replaced based on the **count** of the true/false values within geographic groupings. (Since it's a boolean feature)

### New features created
- year_recorded
    This feature is created using the _date_recorded_ feature which represents the date, the waterpump details are recorded. Since the majority of dates are closely related based on the year, this feature is created.

    | unique year | value count |
    | --- | --- |
    2011 | 28674
    2013 | 24271
    2012 | 6424
    2004 | 30
    2002 | 1

### Categorical Feature Encoding
* region
* payment_type
* water_quality
* extraction_type_class
* waterpoint_type
* basin
* source
* quantity_group

Tested the performance with three types of encoding

#### 1. Random Encoding
Used Pandas `factorize()` method to encode.
Random integer values are assigned within the range [0, num_categories - 1]

#### 2. Frequency Encoding
Encoded the categories of a feature based on the value counts of each category

#### 3. Target Encoding
Encoded the categories of a feature based on the ratio of `functional` count against the total count

    Example: Target encoding for quantity_group feature

![quantity_group_countplot](assets\quantity_group_countplot.png)

According to above count-plot, `'enough'` category obtains the highest target encoding value and `'dry'` category obtains the lowest target encoding value.

## Model

**Random Forest Calssifier** is used with 400 decision-trees.

5-fold Cross Validation is used with scoring metrics: `'precision_macro'` and `'recall_macro'`

Best accuracy at the competition obtained from the **target encoding** approach
    
    Accuracy: 80.95%

![accuracy](assets\accuracy.png)