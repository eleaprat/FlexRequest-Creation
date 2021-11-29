# FlexRequest-Creation
The code is associated with the paper [*Network-Aware Flexibility Requests for Distribution-Level Flexibility Markets*](https://arxiv.org/abs/2110.05983) by E. Prat et al.

## Instructions for running

### Stochastic Market Clearing
From the folder **Stochastic Clearing**, run **stoch_MC_main.py**. The following must be done before running:
  * Copy the folder **Wind Data** to the folder **Stochastic Clearing**
  * Copy the folder **15bus Data** to the folder **Stochastic Clearing**
  * Update the necessary parameters
The results can later be found in **Stochastic Clearing/15bus Data/Results**.

### FlexRequest Creation
From the folder **FlexRequest Creation**, run **FR_main.py**. The following must be done before running:
  * Copy the folder **Wind Data** to the folder **FlexRequest Creation**
  * Copy the folder **15bus Data** to the folder **FlexRequest Creation**
  * Update the necessary parameters
The results can later be found in **FlexRequest Creation/15bus Data/Results**.

### Deterministic Market Clearing
From the folder **Deterministic Clearing**, run **FR_main**. The following must be done before running:
  * Run **FR_main.py**
  * From **FlexRequest Creation/15bus Data/Results** retrieve the csv file with the resulting requests and copy it to **Deterministic Clearing/15bus Data**
  * Copy the folder **15bus Data** to the folder **Deterministic Clearing**
  * Update the necessary parameters
The results can later be found in **Deterministic Clearing/15bus Data/Results**.

### Real-Time Verification
From the folder **Real Time**, run **RT_main.py**. The following must be done before running:
  * Run the Stochastic Market Clearing
  * From **Stochastic Clearing/15bus Data/Results** retrieve the csv files with the accepted bids and copy them to **Real Time/15bus Data**
  * Run the Deterministic Market Clearing
  * From **Deterministic Clearing/15bus Data/Results** retrieve the csv files with the accepted bids and copy them to **Real Time/15bus Data**
  * Copy the folder **Wind Data** to the folder **Real Time**
  * Copy the folder **15bus Data** to the folder **Real Time**
  * Update the necessary parameters
The results can later be found in **Real Time/15bus Data/Results**.
