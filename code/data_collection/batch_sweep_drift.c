/*----------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------

            Copyright (c) 2020 - 2021 Keysight Technologies Inc.

All Rights Reserved. Reproduction, adaptation, or translation without prior
written permission is prohibited, except as allowed under the copyright laws.
PROPRIETARY RIGHTS of Keysight Technologies are involved in the subject matter
of this material. All manufacturing, reproduction, use, and sales rights
pertaining to this subject matter are governed by the license agreement.
The recipient of this software implicitly accepts the terms of the license.

 * The program is a comprehensive C example illustrating the following sequence
 *    - Read a local configuration file (config.txt) to determine the most recent connection
        configuration, forward and reflected gain settings, muti-unit factory and/or user calibration table
        constructed state and sweep setup.
 *    - Determines if user wants to run the program with the most recent configuration or re-setup
        unit configurations
        NOTE: it is recommended to re-setup if the connected units have been replaced/changed
      - Lets user setup the connected unit configurations, forward and reflected gain and construct
        multi-unit factory and/or user calibration table if they choose not to run with previous setup
 *    - Connects to the MN7021A units with the decided configuration and retrieves device information of
        each connected unit
 *    - Remind users on the required connection configuration for LO and Reference of the multi-unit setup
 *    - Checks reference clock detection, reference lock and health status of each unit
 *    - Determines if user wants to enable multi-unit factory calibration factors / user calibration factors
 *    - Determines if user wants to run sweep with the most recent detected sweep setup
 *    - Lets user setup sweep parameters if they choose not to run with previous setup
 *    - Determine if user wants to wait for thermal stabilization before proceeding for calibration
 *    - Performs sweep
 *    - Determines if user wants to repeat the sweep
 *    - Exit program
 *
 * NOTE: Result file is produced in the /Data folder
 *    - SPARAM_ReArr_<timestamp>_FreqSweep_ReIm.csv : S-parameters result in Real-Imaginary format utilizing Frequency Sweep method
 *    - SPARAM_ReArr_<timestamp>_FreqSweep_MagPhs.csv : S-parameters result in Magnitude-Phase format utilizing Frequency Sweep method
 *    - SPARAM_ReArr_<timestamp>_PortSweep_ReIm.csv : S-parameters result in Real-Imaginary format utilizing Port Sweep method
 *    - SPARAM_ReArr_<timestamp>_PortSweep_MagPhs.csv : S-parameters result in Magnitude-Phase format utilizing Port Sweep method

------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "MN7021aApp.h"
#include <complex.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>
#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <libxml2/libxml/parser.h>
#include <libxml2/libxml/xmlreader.h>
#include <signal.h>

#define STORAGE_ID1 "/SHM_SPARAM1"
#define STORAGE_ID2 "/SHM_SPARAM2"

#define _LP2_  1

/**
	Global variables
**/

char *SegmentConfigFile = "SegmentConfig.csv";
const char * ifTable[13] = { "6.0MHz","3.0MHz","1.5MHz","750kHz","390kHz",
		                    "195kHz","100kHz"," 50kHz"," 25kHz"," 12kHz"," 6kHz"," 3kHz","1.5kHz"};
const IFBandwidth ifSetting[13] ={IFBW_6MHz, IFBW_3MHz, IFBW_1p5MHz, IFBW_750kHz, IFBW_390kHz,
                                IFBW_195kHz, IFBW_100kHz, IFBW_50kHz, IFBW_25kHz, IFBW_12kHz, IFBW_6kHz,
                                IFBW_3kHz, IFBW_1p5kHz};
const SweepMode swMode[2] = {MODE0, MODE1};
const int saveModeArr[4] = {SAVETONONE, SAVETOMEM, SAVETOFILE, SAVETOMEMANDFILE};

int8_t fileNumOfHunter; // read config.txt saved number of MN7021A connected from the most recent program execution
char fileHunterSerNum [4][8]; // read config.txt saved serial numbers of connected MN7021A from the most recent program execution
char fileBoardNum [4];  // read config.txt saved assigned board numbers of MN7021A from the most recent program execution
int socket_desc;
int client_sock;
FactCalFactors testFactoryMultiCalFact [MAX_FILE_POINTS][16][16]; // to store multi-unit factory calibration factors
UserCalFactors testUserMultiCalFact [MAX_FILE_POINTS][16][16]; // to store multi-unit user calibration factors
UserThruFactors testThruCalFact [MAX_FILE_POINTS][16][16]; // to store through calibration factors
double SParam1shm[MAX_SWEEP_COUNT][CHAN_DATA_LEN]; // Sweep results S-parameter Magnitude component
double SParam2shm[MAX_SWEEP_COUNT][CHAN_DATA_LEN]; // Sweep results S-parameter Phase component

/**
	Local function prototype
**/
int GetCalInput(int* ans);
int GetGainInput(char* ans);
const char* GetIFBWEnumStringFromEnum(unsigned short bw);
const char* GetIFBWEnumStringFromUInt16(unsigned short bw);
bool CompareAllUnitFrimwareVer(int devCount, char allFwVer[4][8]);
int GetChoice(int* ans);
void CheckSweepPoints(double* start, double* stop, double* step);

void lossCommunicationSignalHandler(int signum);
void sig_handler(int signum);
void *AbortSweepIntrpt(void* arg);

void got_alarm(int sig);
void sig_handler2(int signum);
void ComputeRecSupvParam(int sweepCount, int* intvlmicroS);
void GetInputDataCorr(int rStatus);


    pthread_t parent;
    // bool abortTrig = false;

// ================================ Main program =====================================


// --- Forward declarations for batch_sweep helpers (definitions at end) ---
#define BS_MAX_SAVED 64
#define BS_NAME_LEN  128
#define BS_MAX_SKIPS 256
#define BS_LABEL_LEN 24
#define BS_CONFIG_DIR "./batch_configs"

// How many antennas the user said they're using (1-4). Set in main() during
// session setup. After bs_write_csv writes the full 4-port CSV, we post-
// process it to keep only the S-parameter columns relevant to the antennas
// in use. Ports 1..g_bs_numAntennas are kept; the rest are stripped.
static int g_bs_numAntennas = 4;

void bs_read_line(const char *prompt, char *out, size_t outSize);
int  bs_read_int(const char *prompt, int defaultVal);
double bs_read_double(const char *prompt, double defaultVal);
int  bs_read_int_list(const char *prompt, int *out, int maxCount);
void bs_sanitize(const char *in, char *out, size_t outSize);
void bs_ensure_dir(const char *path);
void bs_grid_to_physical(int row, int col, int subPos,
                         double cellSize, double dividerThick,
                         double *xInch, double *yInch);
void bs_write_metadata(const char *path, const char *modelName,
                       const char *antennaName, const char *objectName,
                       const char *operatorName, int gridRows, int gridCols,
                       int *measureRows, int numMeasureRows,
                       int *measureCols, int numMeasureCols,
                       double cellSizeInch, double dividerInch,
                       int trialCount, int numPositions, int autoMode,
                       const char *notes);
void bs_write_readme(const char *path, const char *modelName,
                     const char *antennaName, const char *objectName,
                     const char *operatorName, int gridRows, int gridCols,
                     int *measureRows, int numMeasureRows,
                     int *measureCols, int numMeasureCols,
                     double cellSizeInch, double dividerInch,
                     int trialCount, int numPositions, int autoMode,
                     int positionsTaken, int positionsSkipped,
                     char sessionSkips[][BS_LABEL_LEN], int numSessionSkips,
                     const char *sessionFolderName,
                     const char *notes);
void bs_run_one_sweep(int numOfDevices, char swpTypeInput, char *unitNumber,
                      int *socketsArranged, double *startF, double *stopF,
                      double *stepF, IFBandwidth ifBw, int resultFormat,
                      int swpCnt, SweepMode mode, int delayBtwSwp,
                      int saveMode, int segSwp, int *segPort_arg,
                      bool avgSwpData, int *fd1, int *fd2,
                      pthread_t *abortThrId);
void bs_write_csv(const char *csvPath, int numOfDevices, int numPoints,
                  int swpCnt, int resultFormat, int segSwp, bool avgSwpData);
void bs_filter_csv_to_antennas(const char *csvPath, int numAntennas);

// Persistent config helpers
int  bs_load_list(const char *filename, char items[][BS_NAME_LEN], int maxItems);
void bs_save_list(const char *filename, char items[][BS_NAME_LEN], int count);
int  bs_append_unique(const char *filename, const char *item);
int  bs_load_grid_config(const char *modelName, int *gr, int *gc,
                         int *measureRows, int *nMR, int *measureCols, int *nMC,
                         double *cellSize, double *divider);
void bs_save_grid_config(const char *modelName, int gr, int gc,
                         int *measureRows, int nMR, int *measureCols, int nMC,
                         double cellSize, double divider);
int  bs_load_skip_list(const char *modelName, char skips[][BS_LABEL_LEN], int maxSkips);
void bs_save_skip_list(const char *modelName, char skips[][BS_LABEL_LEN], int count);
void bs_merge_and_save_skips(const char *modelName,
                             char existingSkips[][BS_LABEL_LEN], int nExisting,
                             char newSkips[][BS_LABEL_LEN], int nNew);
int  bs_is_in_skip_list(const char *label, char skips[][BS_LABEL_LEN], int n);
int  bs_menu_pick(const char *category,
                  const char **builtins, int nBuiltins,
                  char saved[][BS_NAME_LEN], int nSaved,
                  char *result, size_t resultSize, int *wasNew);


int main(void)
{
  
  PerFlags persistFlags;
  short loopcount = 0;
  MN7021aErrType rStatus;
  char SweepParamErrString[200];
  int numOfDevices = 1; // stores input from user on number of connected MN7021A units in the system
  char tempSerialNumbers[40]; // stores serial numbers of connected MN7021A units in the system temporarily
  char serialNumbers[40]; // stores serial numbers of connected MN7021A units in the system
  char unitSerialNumber[10]; // stores queried serial number of single MN7021A unit
  unsigned char unitFirmwareVer[10]; // stores queried firmware version of single MN7021A unit
  char allUnitFirmwareVer[4][8]; // stores queried firmware version of all MN7021A units
  float unitTemperature; // stores queried temperature of single MN7021A unit
  int deviceSocket[4]; // stores the socket number of established connections on each MN7021A units
  int socketsArranged[4]; // stores rearranged socket number according to sequence of assigned board numbers
  int socket;
  char unitNumber[4] = {0}; // stores the assigned or read back board number of each connected MN7021A units in the system
  int primarySocket; // stores the socket number of the connected MN7021a unit which is set as Primary
  char primarySerialNumber[10]; // stores the serial number of the MN7021A unit set as Primary
  char fwdGain[16]; // stores the most recent forward preamp gain setting of each connected MN7021A units in the system
  char reflGain[16]; // stores the most recent reflected preamp gain setting of each connected MN7021A units in the system
  char fGain; // stores forward preamp gain setting to configure
  char rGain; // stores reflected preamp gain setting to configure
  bool extClkConfigured = false;
  bool loCableCal = false;
  float delayTable[4][4] = {{0.0},{0.0}};
  char *CustomSegmentConfigFile = (char *)malloc(256 * sizeof(char));

//   int UknownAlgorithm = 0;  //Disable unless the UnkownThru is enabled.
  unsigned int lockStat = 0;
  unsigned int systemStat = 0;
  bool allLocked = true;
  float powerLevel = 8.0;
  double powerFrequency = 1e9;
  unsigned char powerPortSet[16];

  float MaxTempDelta = 0.50; // maximum allowable temperature delta to consider as achieving thermal stabilization
  float MaxWarmTime = 5.0; //Synth warm up time max
  bool warmupWait = false;
//   float DeltaTemp=5.0,MaxTemp=90.0;
  unsigned int statRes = 0;
  bool cfgChgDetected = false;
  statRes = statRes;
  lockStat = lockStat;
  cfgChgDetected = cfgChgDetected;

  int swpType = 0; // stores input from user on the sweep type to execute [0-Frequency Sweep / 1-Port Sweep]
  int segSwp = 0; // 0 is disable & 1 to enable segment sweep
  int segChannel = 0;
  unsigned short int segActivePort = 0;
  unsigned short int segCheck = 0x0001;
  int segMin = 0;
  int segMax = 0;
  int segCh;
  char swpTypeInput; // sweep type passed as argument into sweep function
  double startF = 500000000.0; // default start sweep frequency
  double stopF = 2000000000.0; // default stop sweep frequency
  double stepF = 4000000.0; // default sweep
  int swpCnt = 1; // sweep count
  int maxPower = 255; // Default power level for segmented config file
  SweepMode mode = MODE0; // default sweep mode
  int modeIndex = 0;
  int delayBtwSwp = 1000; // default delay between sweep at 1000us (only applicable for mode = MODE1;
  int saveMode = 2; // save mode for sweep results storage
  int saveModeIdx = 2; // index for save mode array for sweep results storage
//  bool allUnitPLLic = true;
  int complete = 1;
  int resultFormat = MAG_PHASE;

  IFBandwidth ifBw = 32;
  int ifBwExp = 5;  // Power of 2 for setting the IF Bandwidth

  int counter = 0;
  bool fwDiff = false;
  fwDiff = fwDiff;

  char iface[100] = "eth0"; // Default ethernet interface name

  // to store versions of base driver during program execution
  BASE_DRV_VER VER_s;
  VER_s.Major = 0;
  VER_s.Minor = 0;
  VER_s.Build = 0;
  VER_s.Release = 0;

  // to store versions of application driver during program execution
  APP_DRV_VER APPVER_s;
  APPVER_s.Major = 0;
  APPVER_s.Minor = 0;
  APPVER_s.Build = 0;
  APPVER_s.Release = 0;

  int i = 0;
  int j = 0;
  int k = 0;

  char input; // input from user
  char boardInput = 0; // store input from user for board number assignment
//   char gainInput = 0; // store input from user for gain setting
  int calInput = 0; // store input from user for multi unit cal factor selection
  int thruCalInput = 0;

  int verdict; // flag to check if input from user is valid
  // file descriptors for reading most recent configuration and writing current configuration
  FILE * fpr;
  FILE * fpw;
  FILE * fpw2;
  FILE* testfPtr;
  char* ptr; // pointer variable used in association with reading file
  char * line = NULL; // variable to store read file lines
  size_t len = 0; // variables used in getline() function
  size_t read; // variables used in getline() function
  int lineCnt = 0; // counter used to determine at which line the file read is at
  int readFreq = 0; // flag to indicate that the file read starts reading in frequency parameters for sweep
  int readApCal = 0; // flag to indicate that the file read starts reading in applied calibration
  char boardNum; // to store assigned board number of each board
  bool multiFactoryCal = false; // to store current existence state of multi-unit Factory Cal Factors table
  bool multiUserCal = false; // to store current existence state of multi-unit User Cal Factors table
  bool extClkWritten = false;
  bool loCalWritten = false;
  bool thruCalWritten = false;
  bool thruCal = false; // to store current existence state of User Through Cal Factors table
  bool unkThruCal = false; // to store current existence state of User Unknown Through Cal Factors table
  bool avgSwpData = false; // to store averaging mode flag of sweep data
  char dummy;
  bool exitApp = false;
  ADCcalComplete = false;
  int fd1, fd2;

  extClkWritten = extClkWritten;
  loCalWritten = loCalWritten;
  thruCalWritten = thruCalWritten;

  dummy = ' ';
  dummy = dummy;

  char *pos;
  int appCal;

  CalApplied fileAppliedCal;

  fileAppliedCal.FactCal = false;
  fileAppliedCal.UserCal = false;
  fileAppliedCal.ThruCal = false;
  fileAppliedCal.UnkThruCal = false;
  fileAppliedCal.PortExt = false;
  fileAppliedCal.PortExt = false;

  bool reOptimize = true;

  watchdogTrig = false;
  enableParserLog = false;


//  CalKit CalKits[NUMOFKITS];


  unsigned char SweepCtrlType = 1; // SweepCtrlType = 0 = Standard Sweep. 1 = Licensed Fast Sweep

  bool systemInit = false;

  printf("Standard Sweep Program\n");

  // =============================================================
  // BATCH SWEEP v2 session setup (persistent configs)
  // =============================================================
  char bs_modelName[BS_NAME_LEN]    = {0};
  char bs_antennaName[BS_NAME_LEN]  = {0};
  char bs_objectName[BS_NAME_LEN]   = {0};
  char bs_operatorName[BS_NAME_LEN] = {0};
  int  bs_gridRows = 6, bs_gridCols = 6;
  int  bs_measureRows[64], bs_numMeasureRows = 0;
  int  bs_measureCols[64], bs_numMeasureCols = 0;
  double bs_cellSizeInch = 1.0;
  double bs_dividerInch  = 0.25;
  int  bs_trialCount = 16;
  int  bs_numAntennas = 4;
  int  bs_autoMode = 0;
  double bs_autoDelaySec = 0.2;
  int  bs_driftMode = 0;              // fully unattended drift-test mode
  double bs_driftMoveDelaySec = 3.0;  // simulated per-position "operator move" delay
  char bs_sessionFolder[512] = {0};
  char bs_notes[512] = {0};

  // Skip-tracking state for the position loop
  char bs_autoSkipList[BS_MAX_SKIPS][BS_LABEL_LEN];
  int  bs_numAutoSkips = 0;
  int  bs_useAutoSkip = 0;
  char bs_sessionSkips[BS_MAX_SKIPS][BS_LABEL_LEN];
  int  bs_numSessionSkips = 0;

  // Saved lists (loaded from disk)
  char bs_savedAntennas[BS_MAX_SAVED][BS_NAME_LEN];
  char bs_savedObjects[BS_MAX_SAVED][BS_NAME_LEN];
  char bs_savedPhantoms[BS_MAX_SAVED][BS_NAME_LEN];
  char bs_savedCustomModels[BS_MAX_SAVED][BS_NAME_LEN];

  bs_ensure_dir(BS_CONFIG_DIR);
  int bs_nA = bs_load_list("antennas.list",      bs_savedAntennas,     BS_MAX_SAVED);
  int bs_nO = bs_load_list("objects.list",       bs_savedObjects,      BS_MAX_SAVED);
  int bs_nP = bs_load_list("phantoms.list",      bs_savedPhantoms,     BS_MAX_SAVED);
  int bs_nM = bs_load_list("custom_models.list", bs_savedCustomModels, BS_MAX_SAVED);

  printf("=================================================\n");
  printf("   BATCH SWEEP - Imager Data Recording (v2)\n");
  printf("=================================================\n");
  printf("Saved: %d antennas, %d objects, %d phantom versions, %d custom models\n\n",
         bs_nA, bs_nO, bs_nP, bs_nM);

  // ----- Antenna selection -----
  const char *bs_builtinAntennas[] = {"Medium Hoof Antenna", "McGill Antenna"};
  int bs_wasNew = 0;
  bs_menu_pick("Antenna", bs_builtinAntennas, 2,
               bs_savedAntennas, bs_nA,
               bs_antennaName, sizeof(bs_antennaName), &bs_wasNew);
  if (bs_wasNew) {
    char saveResp[16];
    bs_read_line("Save this antenna for future use? [y/N]: ", saveResp, sizeof(saveResp));
    if (saveResp[0] == 'y' || saveResp[0] == 'Y') {
      if (bs_append_unique("antennas.list", bs_antennaName))
        printf("  Antenna saved.\n");
    }
  }

  // ----- Object selection (no built-ins) -----
  bs_menu_pick("Object", NULL, 0,
               bs_savedObjects, bs_nO,
               bs_objectName, sizeof(bs_objectName), &bs_wasNew);
  if (bs_wasNew) {
    char saveResp[16];
    bs_read_line("Save this object for future use? [y/N]: ", saveResp, sizeof(saveResp));
    if (saveResp[0] == 'y' || saveResp[0] == 'Y') {
      if (bs_append_unique("objects.list", bs_objectName))
        printf("  Object saved.\n");
    }
  }

  // ----- Model selection (with sub-menu for Breast Phantom) -----
  const char *bs_builtinTopModels[] = {"ButterBox", "Breast Phantom"};
  char bs_topModel[BS_NAME_LEN] = {0};
  bs_menu_pick("Model", bs_builtinTopModels, 2,
               bs_savedCustomModels, bs_nM,
               bs_topModel, sizeof(bs_topModel), &bs_wasNew);

  if (strcmp(bs_topModel, "Breast Phantom") == 0) {
    // Sub-menu for phantom version
    char bs_phantomVer[BS_NAME_LEN] = {0};
    int bs_phantomNew = 0;
    bs_menu_pick("Breast Phantom version",
                 NULL, 0,
                 bs_savedPhantoms, bs_nP,
                 bs_phantomVer, sizeof(bs_phantomVer), &bs_phantomNew);
    if (bs_phantomNew) {
      char saveResp[16];
      bs_read_line("Save this phantom version for future use? [y/N]: ", saveResp, sizeof(saveResp));
      if (saveResp[0] == 'y' || saveResp[0] == 'Y') {
        if (bs_append_unique("phantoms.list", bs_phantomVer))
          printf("  Phantom version saved.\n");
      }
    }
    snprintf(bs_modelName, sizeof(bs_modelName), "BreastPhantom_%s", bs_phantomVer);
  } else {
    strncpy(bs_modelName, bs_topModel, sizeof(bs_modelName) - 1);
    if (bs_wasNew) {
      char saveResp[16];
      bs_read_line("Save this model for future use? [y/N]: ", saveResp, sizeof(saveResp));
      if (saveResp[0] == 'y' || saveResp[0] == 'Y') {
        if (bs_append_unique("custom_models.list", bs_modelName))
          printf("  Model saved.\n");
      }
    }
  }
  printf("\nModel: %s\n\n", bs_modelName);

  // ----- Grid config (load saved or prompt new) -----
  int bs_haveSavedGrid = bs_load_grid_config(bs_modelName,
                                              &bs_gridRows, &bs_gridCols,
                                              bs_measureRows, &bs_numMeasureRows,
                                              bs_measureCols, &bs_numMeasureCols,
                                              &bs_cellSizeInch, &bs_dividerInch);
  int bs_useSavedGrid = 0;
  if (bs_haveSavedGrid) {
    printf("Saved grid for model '%s':\n", bs_modelName);
    printf("  Grid: %dx%d\n", bs_gridRows, bs_gridCols);
    printf("  Measured rows:");
    for (int ii = 0; ii < bs_numMeasureRows; ii++) printf(" %d", bs_measureRows[ii]);
    printf("\n  Measured cols:");
    for (int ii = 0; ii < bs_numMeasureCols; ii++) printf(" %d", bs_measureCols[ii]);
    printf("\n  Cell size: %.3f in,  Divider: %.3f in\n", bs_cellSizeInch, bs_dividerInch);
    char useResp[16];
    bs_read_line("Use these saved grid settings? [Y/n]: ", useResp, sizeof(useResp));
    bs_useSavedGrid = !(useResp[0] == 'n' || useResp[0] == 'N');
  } else if (strcmp(bs_modelName, "ButterBox") == 0) {
    // ButterBox built-in defaults: 6x6 grid, inner 4x4 measured
    printf("ButterBox default grid: 6x6, measuring rows 2-5 cols 2-5, 1.0\" cells, 0.25\" dividers.\n");
    char useResp[16];
    bs_read_line("Use these ButterBox defaults? [Y/n]: ", useResp, sizeof(useResp));
    if (!(useResp[0] == 'n' || useResp[0] == 'N')) {
      bs_gridRows = 6; bs_gridCols = 6;
      bs_numMeasureRows = 4; bs_numMeasureCols = 4;
      for (int i = 0; i < 4; i++) { bs_measureRows[i] = i + 2; bs_measureCols[i] = i + 2; }
      bs_cellSizeInch = 1.0; bs_dividerInch = 0.25;
      bs_useSavedGrid = 1;
    }
  }

  if (!bs_useSavedGrid) {
    bs_gridRows      = bs_read_int   ("Total grid rows: ", 6);
    bs_gridCols      = bs_read_int   ("Total grid columns: ", 6);
    bs_numMeasureRows = bs_read_int_list("Row numbers to MEASURE (space-separated): ",
                                         bs_measureRows, 64);
    bs_numMeasureCols = bs_read_int_list("Column numbers to MEASURE (space-separated): ",
                                         bs_measureCols, 64);
    bs_cellSizeInch  = bs_read_double("Cell size in inches [1.0]: ", 1.0);
    bs_dividerInch   = bs_read_double("Divider thickness in inches [0.25]: ", 0.25);

    char saveResp[16];
    bs_read_line("Save this grid config for this model? [Y/n]: ", saveResp, sizeof(saveResp));
    if (!(saveResp[0] == 'n' || saveResp[0] == 'N')) {
      bs_save_grid_config(bs_modelName, bs_gridRows, bs_gridCols,
                          bs_measureRows, bs_numMeasureRows,
                          bs_measureCols, bs_numMeasureCols,
                          bs_cellSizeInch, bs_dividerInch);
      printf("  Grid config saved for '%s'.\n", bs_modelName);
    }
  }

  // ----- Skip list (load if exists, ask if auto-skip) -----
  bs_numAutoSkips = bs_load_skip_list(bs_modelName, bs_autoSkipList, BS_MAX_SKIPS);
  if (bs_numAutoSkips > 0) {
    printf("\nSaved skip list for '%s' (%d positions):\n", bs_modelName, bs_numAutoSkips);
    for (int ii = 0; ii < bs_numAutoSkips; ii++) printf("  %s\n", bs_autoSkipList[ii]);
    char useResp[16];
    bs_read_line("Auto-skip these positions this session? [Y/n]: ", useResp, sizeof(useResp));
    bs_useAutoSkip = !(useResp[0] == 'n' || useResp[0] == 'N');
  }

  // ----- Operator and trial count -----
  bs_read_line("\nOperator name: ", bs_operatorName, sizeof(bs_operatorName));
  bs_trialCount = bs_read_int("Trials per position [16]: ", 16);

  // ----- Number of antennas in use -----
  // Each saved CSV will only include the S-parameters that involve the active
  // antennas. 2 antennas = ports 1 & 2 (S11, S12, S21, S22 only). 3 antennas
  // = ports 1, 2, 3 (S11..S33). 4 antennas = full 4-port matrix.
  bs_numAntennas = bs_read_int("Number of antennas in use (1-4) [4]: ", 4);
  if (bs_numAntennas < 1) bs_numAntennas = 1;
  if (bs_numAntennas > 4) bs_numAntennas = 4;
  g_bs_numAntennas = bs_numAntennas;
  printf("  Recording S-parameters for ports 1..%d only.\n", bs_numAntennas);

  // ----- Recording mode -----
  {
    char modeBuf[16] = {0};
    bs_read_line("Recording mode - (A)utomatic, (I)nteractive, or (D)rift-test [I]: ",
                 modeBuf, sizeof(modeBuf));
    if (modeBuf[0] == 'D' || modeBuf[0] == 'd') {
      // Fully unattended drift-test mode. Same trial timing as (A)utomatic,
      // PLUS a configurable inter-position delay to simulate an operator
      // moving an object. No prompts. No object is really placed - phantom
      // stays empty throughout, so the recorded S-parameters are all
      // "empty-phantom" and any variation IS drift.
      bs_driftMode = 1;
      bs_autoMode = 1;   // drift mode implies automatic trial sweeps
      bs_autoDelaySec = bs_read_double(
        "Delay between trial sweeps in seconds [0.2]: ", 0.2);
      bs_driftMoveDelaySec = bs_read_double(
        "Simulated inter-position delay in seconds [3.0]: ", 3.0);
      printf("  DRIFT-TEST MODE: fully unattended. All prompts auto-confirmed.\n");
      printf("  Trial spacing = %.2fs   Inter-position delay = %.2fs\n",
             bs_autoDelaySec, bs_driftMoveDelaySec);
    } else if (modeBuf[0] == 'A' || modeBuf[0] == 'a') {
      bs_autoMode = 1;
      bs_autoDelaySec = bs_read_double("Delay between auto sweeps in seconds [0.2]: ", 0.2);
    } else {
      bs_autoMode = 0;
    }
  }

  // ----- Additional notes (free-form, optional) -----
  // Anything the user wants to record about this session: antenna config,
  // cable layout, environmental conditions, etc. Gets saved verbatim into
  // session_metadata.txt and the per-session README.md. Empty is fine.
  bs_read_line("\nAdditional notes (antenna config, conditions, etc.;\n  or just press Enter to skip): ",
               bs_notes, sizeof(bs_notes));

  int bs_numPositions = bs_numMeasureRows * bs_numMeasureCols * 4;

  // ----- Build session folder -----
  {
    time_t bs_now = time(NULL);
    struct tm bs_tm = *localtime(&bs_now);
    char bs_modelClean[BS_NAME_LEN], bs_objectClean[BS_NAME_LEN];
    bs_sanitize(bs_modelName,  bs_modelClean,  sizeof(bs_modelClean));
    bs_sanitize(bs_objectName, bs_objectClean, sizeof(bs_objectClean));
    snprintf(bs_sessionFolder, sizeof(bs_sessionFolder),
             "./Data/%s_%s_%04d%02d%02d_%02d%02d",
             bs_modelClean, bs_objectClean,
             bs_tm.tm_year + 1900, bs_tm.tm_mon + 1, bs_tm.tm_mday,
             bs_tm.tm_hour, bs_tm.tm_min);
    bs_ensure_dir("./Data");
    bs_ensure_dir(bs_sessionFolder);
  }

  printf("\n-------------------------------------------------\n");
  printf("Session folder: %s\n", bs_sessionFolder);
  printf("Positions: %d   Trials/pos: %d   Total sweeps: %d\n",
         bs_numPositions, bs_trialCount, bs_numPositions * bs_trialCount);
  printf("Mode: %s   Auto-skip: %s\n",
         bs_driftMode ? "DRIFT-TEST" : (bs_autoMode ? "AUTOMATIC" : "INTERACTIVE"),
         bs_useAutoSkip ? "yes" : "no");
  if (bs_driftMode) {
    printf("Drift-test timing: %.2fs / trial, %.2fs / position\n",
           bs_autoDelaySec, bs_driftMoveDelaySec);
  }
  printf("-------------------------------------------------\n");
  if (bs_driftMode) {
    printf("[DRIFT MODE] Auto-continuing to VNA setup (no operator gate).\n");
  } else {
    printf("After Keysight setup completes you'll be prompted at each position.\n");
    printf("Press Enter to continue with VNA setup...\n");
    { int c; while ((c = getchar()) != '\n' && c != EOF) {} }
  }

  // Write session metadata up front
  {
    char bs_metaPath[640];
    snprintf(bs_metaPath, sizeof(bs_metaPath), "%s/session_metadata.txt", bs_sessionFolder);
    bs_write_metadata(bs_metaPath, bs_modelName, bs_antennaName, bs_objectName,
                      bs_operatorName, bs_gridRows, bs_gridCols,
                      bs_measureRows, bs_numMeasureRows,
                      bs_measureCols, bs_numMeasureCols,
                      bs_cellSizeInch, bs_dividerInch,
                      bs_trialCount, bs_numPositions, bs_autoMode,
                      bs_notes);
  }
  // =============================================================
  // end BATCH SWEEP v2 session setup
  // =============================================================


  QueryDriverVersions(&VER_s, &APPVER_s);
  printf("[INFO] Shared Library version %d.%d.%d.%d\n", VER_s.Major, VER_s.Minor, VER_s.Build, VER_s.Release);
  printf("[INFO] Application Driver version %d.%d.%d.%d\n\n", APPVER_s.Major, APPVER_s.Minor, APPVER_s.Build, APPVER_s.Release);




    DIR* dir = opendir("./Data");
    if (dir)
    {
        /* Directory exists. */
        closedir(dir);
    }
    else if (ENOENT == errno)
    {
        mkdir("./Data", 0777);
    }
    else
    {
        //
    }

  // check to see if the configuration file is present and proceed to read the file line by line
   rStatus = ValidateConfigFile("config.txt");
    if (rStatus == MN7021aERR_NONE)
    {
        fpr = fopen("config.txt", "r");

    // stores the necessary configuration and sweep setup parameters read from file
    int s = 0;
    int g = 0;
    while ((read = getline(&line, &len, fpr)) != -1)
    {
        if (lineCnt > 0)
        {
            if (strcmp(line, "MOST RECENT SWEEP SETUP\n") == 0)
            {
                read = getline(&line, &len, fpr);
                readFreq = 1;
            }
            else if (strcmp(line, "MOST RECENT GAIN SETTINGS\n") == 0)
            {
                g = 0;

                read = getline(&line, &len, fpr);
                read = getline(&line, &len, fpr);

                ptr = strtok(line, ",");
                while (ptr != NULL)
                {
                    fwdGain[g] = atoi(ptr);
                    ptr = strtok(NULL, ",");
                    g++;
                }

                g = 0;

                read = getline(&line, &len, fpr);

                ptr = strtok(line, ",");
                while (ptr != NULL)
                {
                    reflGain[g] = atoi(ptr);
                    ptr = strtok(NULL, ",");
                    g++;
                }

                g = 0;
            }
            else if (strcmp(line, "FACTORY MULTI CAL:\n") == 0)
            {
                read = getline(&line, &len, fpr);

                if (strcmp(line, "True\n") == 0)
                {
                    multiFactoryCal = true;
                }
                else
                {
                    multiFactoryCal = false;
                }
            }
            else if (strcmp(line, "USER MULTI CAL:\n") == 0)
            {
                read = getline(&line, &len, fpr);

                if (strcmp(line, "True\n") == 0)
                {
                    multiUserCal = true;
                }
                else
                {
                    multiUserCal = false;
                }
            }
            else if (strcmp(line, "CONFIGURED EXT CLOCK:\n") == 0)
            {
                read = getline(&line, &len, fpr);

                if (strcmp(line, "True\n") == 0)
                {
                    extClkConfigured = true;
                }
                else
                {
                    extClkConfigured = false;
                }
            }
            else if (strcmp(line, "LO CABLE CAL:\n") == 0)
            {
                read = getline(&line, &len, fpr);

                if (strcmp(line, "True\n") == 0)
                {
                    loCableCal = true;
                }
                else
                {
                    loCableCal = false;
                }
            }
            else if (strcmp(line, "THROUGH CAL:\n") == 0)
            {
                read = getline(&line, &len, fpr);

                if (strcmp(line, "True\n") == 0)
                {
                    thruCal = true;
                }
                else
                {
                    thruCal = false;
                }
            }
            else if (strcmp(line, "UNKNOWN THROUGH CAL:\n") == 0)
            {
                read = getline(&line, &len, fpr);

                if (strcmp(line, "True\n") == 0)
                {
                    unkThruCal = true;
                }
                else
                {
                    unkThruCal = false;
                }
            }
            else if (strcmp(line, "APPLIED CAL\n") == 0)
            {
                read = getline(&line, &len, fpr);
                readApCal = 1;
            }
            else if ((readFreq > 0) && (readFreq < 11))
            {
//                char *pos;
                if ((pos=strchr(line, '\n')) != NULL)
                {
                    *pos = '\0';
                }

                switch (readFreq)
                {
                    case 1:
                    {
                        swpType = line[0] - '0';
                        break;
                    }

                    case 2:
                    {
                        startF = atof(line);
                        break;
                    }

                    case 3:
                    {
                        stopF = atof(line);
                        break;
                    }

                    case 4:
                    {
                        stepF = atof(line);
                        break;
                    }

                    case 5:
                    {
                        swpCnt = atoi(line);
                        break;
                    }

                    case 6:
                    {
                        ifBw = atoi(line);
                        break;
                    }

                    case 7:
                    {
                        mode = atoi(line);
                        break;
                    }

                    case 8:
                    {
                        delayBtwSwp = atoi(line);
                        break;
                    }

                    case 9:
                    {
                        saveMode = atoi(line);
                        break;
                    }

                    case 10:
                    {
                        segSwp = atoi(line);
                        break;
                    }

                    default:
                        break;
                }

                readFreq++;
            }
            else if ((readApCal > 0) && (readApCal < 7))
            {
//                char *pos;
//                if ((pos=strchr(line, '\n')) != NULL)
//                {
//                    *pos = '\0';
//                }

                appCal = atoi(line);

                switch (readApCal)
                {
                    case 1:
                    {
                        fileAppliedCal.FactCal = (bool)(appCal);
                        break;
                    }

                    case 2:
                    {
                        fileAppliedCal.UserCal = (bool)(appCal);
                        break;
                    }

                    case 3:
                    {
                        fileAppliedCal.ThruCal = (bool)(appCal);
                        break;
                    }

                    case 4:
                    {
                        fileAppliedCal.UnkThruCal = (bool)(appCal);
                        break;
                    }

                    case 5:
                    {
                        fileAppliedCal.PortExt = (bool)(appCal);
                        break;
                    }

                    case 6:
                    {
                        fileAppliedCal.LoCblDly = (bool)(appCal);
                        break;
                    }

                    default:
                        break;
                }

                readApCal++;
            }
            else
            {
                if (strcmp(line, "\n") != 0)
                {
                    for (s = 0; s < 8; s++)
                    {
                        fileHunterSerNum[lineCnt - 1][s] = line[s];
                    }

                    fileBoardNum[lineCnt - 1] =  line[10] - '0';
                }
            }
        }
        else
        {
            fileNumOfHunter = line[17] - '0';
        }

        lineCnt++;
    }


    readFreq = 0;
    fclose(fpr);
    if (line)
    {
        free(line);
    }

    // check if user wants to enable Message logging
    printf("MESSAGE LOGGING\n");
    printf("=========================\n\n");
    printf("Do you want to enable message logging [Y/N]? ");

     while ((verdict = GetInput(&input)) != 0)
    {
        printf("Please input 'Y' or 'y' to enable message logging\n");
        printf("or input 'N' or 'n' to disable message logging: ");
    }

    // Enable message logging
    if ((input == 'Y') || (input == 'y'))
    {
        enableMessageLog = true;
        int option;
        int result;

        do
        {
             // prompt user to select logging level
            printf("\nPlease select the level of logging [1-3]:\n");
            printf("1) LOG_LEVEL_ERROR - Log only error messages \n");
            printf("2) LOG_LEVEL_WARNING - Log only error and warning messages \n");
            printf("3) LOG_LEVEL_INFO - Log everything \n\n");
            printf("Enter your option: ");
            result = scanf("%d", &option);

            if (result != 1)
            {
                while (getchar() != '\n');
                printf("Invalid option! Please select a number between 1 and 3. \n");
                continue;
            }

            if(option < 1 || option > 3)
            {
                printf("Invalid option! Please select a number between 1 and 3. \n");
            }

        } while (option < 1 || option > 3);

        switch(option)
            {
                case 1:
                    printf("LOG_LEVEL_ERROR has been selected. \n\n");
                    set_log_level(LOG_LEVEL_ERROR);
                    break;
                case 2:
                    printf("LOG_LEVEL_WARNING has been selected. \n\n");
                    set_log_level(LOG_LEVEL_WARNING);
                    break;
                case 3:
                    printf("LOG_LEVEL_INFO has been selected. \n\n");
                    set_log_level(LOG_LEVEL_INFO);
                    break;
            }
    }

    // displays the unit configuration read from the configuration file
    printf("MOST RECENT CONFIGURATION\n");
    printf("=========================\n\n");
    printf("Number of connected units: %d\n\n", fileNumOfHunter);

    for (int p = 0; p < fileNumOfHunter; p++)
    {
        for (s = 0; s < 8; s++)
        {
            printf("%c", fileHunterSerNum[p][s]);
        }

        printf(" --->");
        printf("\nAssigned board number: ");
        printf("%d", fileBoardNum[p]);
        printf("\nForward Gain: ");

        boardNum = fileBoardNum[p];

        for (int fg = (boardNum - 1) * 4; fg < ((boardNum - 1) * 4) + 4; fg++)
        {
            printf("%d", fwdGain[fg]);

            if (fg < ((boardNum - 1) * 4) + 3)
            {
                printf(",");
            }
        }

        printf("\nReflected Gain: ");

        for (int rg = (boardNum - 1) * 4; rg < ((boardNum - 1) * 4) + 4; rg++)
        {
            printf("%d", reflGain[rg]);

            if (rg < ((boardNum - 1) * 4) + 3)
            {
                printf(",");
            }
        }

        printf("\n\n");
    }



    printf("Multi-unit FACTORY Calibration Factor Table constructed? ");
    printf("%s\n", multiFactoryCal ? "True" : "False");
    printf("Multi-unit USER Calibration Factor Table constructed? ");
    printf("%s\n", multiUserCal ? "True" : "False");
    printf("Multi-unit external clock configured? ");
    printf("%s\n", extClkConfigured ? "True" : "False");
    printf("LO CABLE DELAY Table constructed? ");
    printf("%s\n", loCableCal ? "True" : "False");
    printf("THROUGH Calibration Factor Table constructed? ");
    printf("%s\n", thruCal ? "True" : "False");
    printf("UNKNOWN THROUGH Calibration Factor Table constructed? ");
    printf("%s\n", unkThruCal ? "True" : "False");


    printf("\n\nDo you want to continue with this configuration [Y/N]? ");

    while ((verdict = GetInput(&input)) != 0)
    {
        printf("Please input 'Y' or 'y' to continue with this configuration\n");
        printf("or input 'N' or 'n' for a different configuration: ");
    }

    // setup unit configuration according to the detected setup from file
    if ((input == 'Y') || (input == 'y'))
      {
          memset(tempSerialNumbers, 0, sizeof(tempSerialNumbers));
          int matchCtr = 0;
          numOfDevices = fileNumOfHunter;
          // Connect to units
          printf("Do you want to enable the Ethernet Inteface loss communication detection mechanisms [Y/N]? \n");
          while ((verdict = GetInput(&input)) != 0)
          {
              printf("Please input 'Y' or 'y' to continue with this configuration\n");
              printf("or input 'N' or 'n' for a different configuration: ");
          }
          if ((input == 'Y') || (input == 'y'))
          {
              printf("Enter the Ethernet Interface name (For example: eth0): ");
              scanf("%s", iface);
              SetEthernetInterfaceName(iface);
          }
          rStatus = ConnectEthernet(numOfDevices, &serialNumbers[0], &deviceSocket[0]);
          if (rStatus != 0)
          {
            printf("[INFO] Exiting the program...\n");
            exit(-1);
          }

          printf("[INFO] Unit License Info: \n");
          rStatus = QueryLicense(numOfDevices, deviceSocket);


          signal(SIGNUM, lossCommunicationSignalHandler);
          printf("\nDo you want to enable the Unit Monitoring [Y/N]? ");
          while ((verdict = GetInput(&input)) != 0)
          {
             //
          }
          if ((input == 'Y') || (input == 'y'))
          {         
            lossCommunicationMonitor(numOfDevices, SIGNUM);
          }


          for (i = 0; i < numOfDevices; i++)
          {
              matchCtr = 0;

              socket = deviceSocket[i];
              // retrieve device information of each connected unit
              // assign board numbers to each unit as according to the detected configuration from file
              rStatus = EthernetGetDeviceInformation(socket, &unitSerialNumber[0], (char*)&unitFirmwareVer[0], &unitTemperature);

              for (j = 0; j < numOfDevices; j++)
              {
                  for (k = 0; k < 8; k++)
                  {
                      if (unitSerialNumber[k] == fileHunterSerNum[j][k])
                      {
                          matchCtr++;
                      }
                  }

                  if (matchCtr == 8)
                  {
                      break;
                  }
                  else
                  {
                      matchCtr = 0;
                  }
              }
			  if (matchCtr == 0)
			  {
				  printf("[ERROR]: Serial Number not found %s but found %s\n",fileHunterSerNum[j-1],unitSerialNumber);
				  exit(-1);
			  }

              for (counter = 0; counter < 8; counter++)
              {
                  allUnitFirmwareVer[i][counter] = unitFirmwareVer[counter];
              }

              unitNumber[i] = fileBoardNum[j];

              rStatus = EthernetAssignBoardNumber(numOfDevices, fileBoardNum[j], socket, false);
          }

          fwDiff = CompareAllUnitFrimwareVer(numOfDevices, allUnitFirmwareVer);

          if (fwDiff)
		  {
			  printf("WARNING! Inconsistent firmware version detected among the connected units...\n");
			  printf("WARNING! Application may not perform as intended in this condition...\n");
			  printf("Please make sure all units are using the same firmware versions.\n");
			 
			printf("Do you want to continue despite the difference [Y/N] ");
			while ((verdict = GetInput(&input)) != 0)
			{
				printf("Please input 'Y' or 'y' to continue despite the error\n");
				printf("or input 'N' or 'n' to exit: ");
			}

			if (input == 'N' || input == 'n')
			{
				//rStatus = EthernetAbortOperation(numOfDevices, deviceSocket);
				rStatus = DisconnectEthernet(numOfDevices, deviceSocket);
				exit(EXIT_FAILURE);
			}
			
			  
		  }

          // query assigned board numbers to connected units and arrange sockets array according to board number sequence
          rStatus = EthernetQueryBoardNumbersArrangeSockets(numOfDevices, &deviceSocket[0], &unitNumber[0], &socketsArranged[0], &tempSerialNumbers[0]);


     // ************************* CHANGE DETECTION & CAL MANAGEMENT (END) ******************************************************************************

          if (numOfDevices != fileNumOfHunter)
          {
              cfgChgDetected = true;
              printf("ALERT!! The number of connected units has changed from %d to %d.\n", fileNumOfHunter, numOfDevices);
          }
          else
          {
            for (i = 0; i < numOfDevices; i++)
            {
                rStatus = EthernetGetDeviceInformation(socketsArranged[i], &unitSerialNumber[0], (char*)&unitFirmwareVer[0], &unitTemperature);

                if (!CompSerial(&fileHunterSerNum[i][0], &unitSerialNumber[0]))
                {
                    cfgChgDetected = true;
                    printf("ALERT!! BOARD%d: Serial number %s does not compare to saved configuration serial number %s\n", i, unitSerialNumber, fileHunterSerNum[i]);
                }
            }
          }

          if (cfgChgDetected)
          {
              printf("ALERT!! Current detected configuration is different from the saved system configuration.\n");
              printf("ALERT!! Acquisition might not be performed correctly and measurements might be incorrect.\n\n");

              printf("[ACTION] Press ENTER to exit application to start fresh and go through a complete re-setup...\n");
              printf("(Note: the existing ""config.txt"" file will be renamed to ""prevConfig.txt"")\n");

              WaitEnterKey();

              rename("config.txt", "prevConfig.txt");
              rStatus = DisconnectEthernet(numOfDevices, deviceSocket);
              exit(0);
          }

          for (i = 0; i < numOfDevices; i++)
          {
            socket = socketsArranged[i];

            for (j = 0; j < 4; j++)
            {
                fGain = fwdGain[(i * 4) + j];
                rGain = reflGain[(i * 4) + j];
                rStatus = EthernetConfigurePreampGains(socket, j + 1, rGain, fGain);
            }
          }

          if (cfgChgDetected)
          {
              printf("ALERT!! Applied calibration factors might be incorrect.\n");
              printf("ALERT!! Please consider re-constructing/re-calibrating all multi-unit calibration factor tables.\n");


              printf("\nDo you want to re-construct the multi-unit FACTORY calibration factor table for these units [Y/N]? ");

              while ((verdict = GetInput(&input)) != 0)
              {
                  printf("Please input 'Y' or 'y' to re-construct the multi-unit FACTORY calibration factor table\n");
                  printf("or input 'N' or 'n' to skip continue with the existing multi-unit FACTORY calibration factor table");
              }

              // setup unit configuration according to the detected setup from file
              if ((input == 'Y') || (input == 'y'))
              {
                  // ******* Build Multi-unit Factory Cal Factors Table ******
                  rStatus = EthernetDownloadCalCoeffAll(numOfDevices, deviceSocket);
                  GetInputDataCorr(rStatus);
                  ConstructMultiUnitFactoryCalTable(numOfDevices);
                  rStatus = PopulateMultiUnitFactoryCalFactStruct(numOfDevices, testFactoryMultiCalFact);
                  if (rStatus != 0)
                    {
                        exit(0);
                    }
                  multiFactoryCal = true;
              }
              else
              {
                    if (multiFactoryCal)
                    {
                      rStatus = PopulateMultiUnitFactoryCalFactStruct(numOfDevices, testFactoryMultiCalFact);
                      if (rStatus != 0)
                        {
                            exit(0);
                        }
                    }
              }

              if (multiUserCal || thruCal)
              {
                  if (multiUserCal && thruCal)
                  {
                      printf("\nDo you want to exit current application and perform multi-unit USER calibration (OPEN, SHORT & LOAD)");
                      printf("\nand THROUGH calibration for these units [Y/N]? ");
                  }
                  else if (multiUserCal)
                  {
                      printf("\nDo you want to exit current application and perform multi-unit USER calibration (OPEN, SHORT & LOAD) for these units [Y/N]? ");
                  }
                  else
                  {
                      printf("\nDo you want to exit current application and perform THROUGH calibration for these units [Y/N]? ");
                  }


                  while ((verdict = GetInput(&input)) != 0)
                  {
                      printf("Please input 'Y' or 'y' to exit current application\n");
                      printf("or input 'N' or 'n' to skip and continue with the existing calibration factor table");
                  }

                  // setup unit configuration according to the detected setup from file
                  if ((input == 'Y') || (input == 'y'))
                  {
                        exit(0);
                  }
              }

          }



     // ************************* CHANGE DETECTION & CAL MANAGEMENT (END) ******************************************************************************

      }
      else // re-setup unit configuration
      {
          j = 0;
          multiFactoryCal = false;
          multiUserCal = false;
          thruCal = false;
		  extClkConfigured = false;
		  loCableCal = false;
          printf("Please input the number of connected units [1 - 4]: ");
          while ((verdict = GetNumOfDev(&numOfDevices)) != 0)
          {
                printf("Please input between 1 to 4");
          }

          printf("Do you want to enable the Ethernet Inteface loss communication detection mechanisms [Y/N]? \n");
          while ((verdict = GetInput(&input)) != 0)
          {
              printf("Please input 'Y' or 'y' to continue with this configuration\n");
              printf("or input 'N' or 'n' for a different configuration: ");
          }
          if ((input == 'Y') || (input == 'y'))
          {
              printf("Enter the Ethernet Interface name (For example: eth0): ");
              scanf("%s", iface);
              SetEthernetInterfaceName(iface);
          }

          rStatus = ConnectEthernet(numOfDevices, &serialNumbers[0], &deviceSocket[0]);
          if (rStatus != 0)
          {
            printf("[INFO] Exiting the program...\n");
            exit(-1);
          }
          printf("[INFO] Unit License Info: \n");
          rStatus = QueryLicense(numOfDevices, deviceSocket);

          signal(SIGNUM, lossCommunicationSignalHandler);
          printf("\nDo you want to enable the Unit Monitoring [Y/N]? ");
          while ((verdict = GetInput(&input)) != 0)
          {
             //
          }
          if ((input == 'Y') || (input == 'y'))
          {
            lossCommunicationMonitor(numOfDevices, SIGNUM);
          }

          printf("Please assign board number for the following units\n");
          printf("==================================================\n\n");

          for (i = 0; i < numOfDevices; i++)
          {
                socket = deviceSocket[i];

                for (j = 0; j < 10; j++)
                {
                    printf("%c", serialNumbers[(i * 10) + j]);
                }

                printf(": ");

                while ((verdict = GetBoardInput(numOfDevices, &boardInput, &unitNumber[0], i)) != 0)
                {
                    if (verdict == -1)
                    {
                        printf("\n\nPlease input between 1 to %d: ", numOfDevices);

                    }

                    if (verdict == -2)
                    {
                        printf("\n\nThe number %d has already been assigned\n", boardInput);
                        printf("Please assign another board number: ");
                    }
                }

                unitNumber[i] = boardInput;

                rStatus = EthernetAssignBoardNumber(numOfDevices, unitNumber[i], socket, true);
          }

          // query assigned board numbers to connected units and arrange sockets array according to board number sequence
          rStatus = EthernetQueryBoardNumbersArrangeSockets(numOfDevices, &deviceSocket[0], &unitNumber[0], &socketsArranged[0], &serialNumbers[0]);


 
     // ************************* GAIN SETTING ******************************************************************************

      printf("\n\nSETTING FORWARD AND REFLECTED PREAMP GAINS\n");
      printf("====================================================================\n\n");
      
      
	  printf("Forward Gain = 26 \n");
	  printf("Reflected Gain = 22 \n");

	  for (i = 0; i < numOfDevices; i++)
	  {
		for (j = 0; j < 4; j++)
		{
			fwdGain[(i * 4) + j] = 26;
			reflGain[(i * 4) + j] = 22;
		}
	  }
        
      for (i = 0; i < numOfDevices; i++)
      {
            socket = socketsArranged[i];

            for (j = 0; j < 4; j++)
            {
                fGain = fwdGain[(i * 4) + j];
                rGain = reflGain[(i * 4) + j];
                rStatus = EthernetConfigurePreampGains(socket, j + 1, rGain, fGain);
            }
      }

     // ************************* GAIN SETTING (END) ******************************************************************************


     // ************************* CHANGE DETECTION & CAL MANAGEMENT ******************************************************************************

     // This part detects changes of the currently used system with the configuration stored in the host
     // If a change is detected, various prompts will be triggered to remind/notify users if a re-calibration or
     // or a re-download of the factory calibration is desired
        if (numOfDevices != fileNumOfHunter)
        {
          cfgChgDetected = true;
          printf("ALERT!! The number of connected units has changed from %d to %d.\n", fileNumOfHunter, numOfDevices);
        }
        else
        {
        for (i = 0; i < numOfDevices; i++)
        {
            rStatus = EthernetGetDeviceInformation(socketsArranged[i], &unitSerialNumber[0], (char*)&unitFirmwareVer[0], &unitTemperature);

            if (!CompSerial(&fileHunterSerNum[i][0], &unitSerialNumber[0]))
            {
                cfgChgDetected = true;
                printf("ALERT!! BOARD%d: Serial number has changed from %s to serial number %s\n", i, fileHunterSerNum[i], unitSerialNumber);
            }
        }
        }

        if (cfgChgDetected)
        {
          printf("ALERT!! Since the system configuration has changed, previous constructed calibration factors might not be valid anymore.\n");
          printf("ALERT!! Please consider re-constructing all multi-unit calibration factor tables.\n");
        }

        printf("\nDo you want to re-construct the multi-unit FACTORY calibration factor table for these units [Y/N]? ");

        while ((verdict = GetInput(&input)) != 0)
        {
          printf("Please input 'Y' or 'y' to re-construct the multi-unit FACTORY calibration factor table\n");
          printf("or input 'N' or 'n' to skip");
        }

        // setup unit configuration according to the detected setup from file
        if ((input == 'Y') || (input == 'y'))
        {
          // ******* Build Multi-unit Factory Cal Factors Table ******
          rStatus = EthernetDownloadCalCoeffAll(numOfDevices, deviceSocket);
          GetInputDataCorr(rStatus);
          ConstructMultiUnitFactoryCalTable(numOfDevices);
          rStatus = PopulateMultiUnitFactoryCalFactStruct(numOfDevices, testFactoryMultiCalFact);
          if (rStatus != 0)
            {
                exit(0);
            }
          multiFactoryCal = true;
        }
        else
        {
            if (multiFactoryCal)
            {
              rStatus = PopulateMultiUnitFactoryCalFactStruct(numOfDevices, testFactoryMultiCalFact);
              if (rStatus != 0)
                {
                    exit(0);
                }
            }
        }

        if (multiUserCal || thruCal)
        {
          if (multiUserCal && thruCal)
          {
              printf("\nDo you want to exit current application and perform multi-unit USER calibration (OPEN, SHORT & LOAD)");
              printf("\nand THROUGH calibration for these units [Y/N]? ");
          }
          else if (multiUserCal)
          {
              printf("\nDo you want to exit current application and perform multi-unit USER calibration (OPEN, SHORT & LOAD) for these units [Y/N]? ");
          }
          else
          {
              printf("\nDo you want to exit current application and perform THROUGH calibration for these units [Y/N]? ");
          }


          while ((verdict = GetInput(&input)) != 0)
          {
              printf("Please input 'Y' or 'y' to exit current application\n");
              printf("or input 'N' or 'n' to skip and continue with the existing calibration factor table");
          }

          if ((input == 'Y') || (input == 'y'))
          {
                exitApp = true;
          }
          else
          {
                exitApp = false;
          }
        }

     // ************************* CHANGE DETECTION & CAL MANAGEMENT (END) ******************************************************************************


          // write unit configuration setup to file (config.txt)
          fpw = fopen("config.txt", "w");

          fprintf(fpw, "Number of units: %d\n", numOfDevices);

          for (i = 0; i < numOfDevices; i++)
          {
              for (j = 0; j < 8; j++)
              {
                  fprintf(fpw, "%c", serialNumbers[(i * 10) + j]);
              }

              fprintf(fpw, ": %d\n", unitNumber[i]);
          }

          fprintf(fpw, "\nFACTORY MULTI CAL:\n");
          fprintf(fpw, "%s\n", multiFactoryCal ? "True" : "False");
          fprintf(fpw, "\nUSER MULTI CAL:\n");
          fprintf(fpw, "%s\n", multiUserCal ? "True" : "False");
          fprintf(fpw, "\nCONFIGURED EXT CLOCK:\n");
          fprintf(fpw, "%s\n", extClkConfigured ? "True" : "False");
          fprintf(fpw, "\nLO CABLE CAL:\n");
          fprintf(fpw, "%s\n", loCableCal ? "True" : "False");
          fprintf(fpw, "\nTHROUGH CAL:\n");
          fprintf(fpw, "%s\n", thruCal ? "True" : "False");
          fprintf(fpw, "\nUNKNOWN THROUGH CAL:\n");
          fprintf(fpw, "%s\n", unkThruCal ? "True" : "False");

          fclose(fpw);

          /// append gain settings and default sweep setup parameters to file (config.txt)
          fpw = fopen("config.txt", "a");

          fprintf(fpw, "\n\nMOST RECENT GAIN SETTINGS\n");
          fprintf(fpw, "=========================\n");

            for (i = 0; i < numOfDevices; i++)
            {
                for (j = 0; j < 4; j++)
                {
                    fGain = fwdGain[(i * 4) + j];
                    fprintf(fpw, "%d", fGain);

                    if ((i < numOfDevices - 1) || (j < 3))
                    {
                        fprintf(fpw, ",");
                    }
                }
            }

            fprintf(fpw, "\n");

            for (i = 0; i < numOfDevices; i++)
            {
                for (j = 0; j < 4; j++)
                {
                    rGain = reflGain[(i * 4) + j];
                    fprintf(fpw, "%d", rGain);

                    if ((i < numOfDevices - 1) || (j < 3))
                    {
                        fprintf(fpw, ",");
                    }
                }
            }

          fprintf(fpw, "\n\nMOST RECENT SWEEP SETUP\n");
          fprintf(fpw, "=======================\n");

          fprintf(fpw, "0\n");
          fprintf(fpw, "500000000\n");
          fprintf(fpw, "2000000000\n");
          fprintf(fpw, "4000000\n");
          fprintf(fpw, "1\n");
          fprintf(fpw, "32\n");
          fprintf(fpw, "0\n");
          fprintf(fpw, "1000\n");
          fprintf(fpw, "2\n");
          fprintf(fpw, "0\n\n");

          fclose(fpw);

          if (exitApp)
          {
              rStatus = DisconnectEthernet(numOfDevices, socketsArranged);
              exit(0);
          }
      }
  }
  else if (rStatus == MN7021aERR_FILE_EMPTY)
  {

      printf("Would you like to re-setup the configuration [Y/N]?");
      while ((verdict = GetInput(&input)) != 0)
      {
          printf("Please input 'Y' or 'y' to continue with resetup configuration\n");
          printf("or input 'N' or 'n' for a exit program: ");
      }
      if ((input == 'Y') || (input == 'y'))
      {
          goto Resetup;
      }
      else
      {
          exit(EXIT_FAILURE);
      }
  }
  else // first time running on a new host (config.txt file not detected)
  {

  Resetup:j = 0;

      printf("\nPlease input the number of connected units [1 - 4]: ");
      while ((verdict = GetNumOfDev(&numOfDevices)) != 0)
      {
            printf("Please input between 1 to 4: ");
      }

      printf("Do you want to enable the Ethernet Inteface loss communication detection mechanisms [Y/N]? \n");
      while ((verdict = GetInput(&input)) != 0)
      {
          printf("Please input 'Y' or 'y' to continue with this configuration\n");
          printf("or input 'N' or 'n' for a different configuration: ");
      }
      if ((input == 'Y') || (input == 'y'))
      {
          printf("Enter the Ethernet Interface name (For example: eth0): ");
          scanf("%s", iface);
          SetEthernetInterfaceName(iface);
      }

      rStatus = ConnectEthernet(numOfDevices, &serialNumbers[0], &deviceSocket[0]);
      if (rStatus != 0)
      {
        printf("[INFO] Exiting the program...\n");
        exit(-1);
      }
      printf("[INFO] Unit License Info: \n");
      rStatus = QueryLicense(numOfDevices, deviceSocket);

      signal(SIGNUM, lossCommunicationSignalHandler);
      printf("\nDo you want to enable the Unit Monitoring [Y/N]? ");
      while ((verdict = GetInput(&input)) != 0)
      {
        //
      }
      if ((input == 'Y') || (input == 'y'))
      {
        lossCommunicationMonitor(numOfDevices, SIGNUM);
      }

      printf("\n\nPLEASE ASSIGN BOARD NUMBER FOR THE FOLLOWING UNITS\n");
      printf("==================================================\n\n");

      for (i = 0; i < numOfDevices; i++)
      {
            socket = deviceSocket[i];

            for (j = 0; j < 10; j++)
            {
                printf("%c", serialNumbers[(i * 10) + j]);
            }

            printf(": ");


            while ((verdict = GetBoardInput(numOfDevices, &boardInput, &unitNumber[0], i)) != 0)
            {
                if (verdict == -1)
                {
                    printf("\n\nPlease input between 1 to %d: ", numOfDevices);

                }

                if (verdict == -2)
                {
                    printf("\n\nThe number %d has already been assigned\n", boardInput);
                    printf("Please assign another board number: ");
                }
            }

            unitNumber[i] = boardInput;

            rStatus = EthernetAssignBoardNumber(numOfDevices, unitNumber[i], socket, true);
      }

      // query assigned board numbers to connected units and arrange sockets array according to board number sequence
      rStatus = EthernetQueryBoardNumbersArrangeSockets(numOfDevices, &deviceSocket[0], &unitNumber[0], &socketsArranged[0], &serialNumbers[0]);

      if (!multiFactoryCal)
      {
          printf("\nDo you want to construct the multi-unit FACTORY calibration factor table for these units [Y/N]? ");

          while ((verdict = GetInput(&input)) != 0)
          {
              printf("Please input 'Y' or 'y' to construct the multi-unit FACTORY calibration factor table\n");
              printf("or input 'N' or 'n' to skip");
          }

          if ((input == 'Y') || (input == 'y'))
          {
              // ******* Build Multi-unit Factory Cal Factors Table ******
              rStatus = EthernetDownloadCalCoeffAll(numOfDevices, deviceSocket);
              GetInputDataCorr(rStatus);
              ConstructMultiUnitFactoryCalTable(numOfDevices);
              rStatus = PopulateMultiUnitFactoryCalFactStruct(numOfDevices, testFactoryMultiCalFact);
              if (rStatus != 0)
                {
                    exit(0);
                }
              multiFactoryCal = true;
          }
      }

      if (multiUserCal || thruCal)
      {
          if (multiUserCal && thruCal)
          {
              printf("\nMulti-unit USER calibration (OPEN, SHORT & LOAD) and THROUGH calibration has not been performed...");
              printf("\nDo you want to exit current application and perform multi-unit USER calibration (OPEN, SHORT & LOAD)");
              printf("\nand THROUGH calibration for these units [Y/N]? ");
          }
          else if (multiUserCal)
          {
              printf("\nMulti-unit USER calibration (OPEN, SHORT & LOAD) has not been performed...");
              printf("\nDo you want to exit current application and perform multi-unit USER calibration (OPEN, SHORT & LOAD) for these units [Y/N]? ");
          }
          else
          {
              printf("\nTHROUGH calibration has not been performed...");
              printf("\nDo you want to exit current application and perform THROUGH calibration for these units [Y/N]? ");
          }


          while ((verdict = GetInput(&input)) != 0)
          {
              printf("Please input 'Y' or 'y' to exit current application\n");
              printf("or input 'N' or 'n' to skip and continue");
          }

          // setup unit configuration according to the detected setup from file
          if ((input == 'Y') || (input == 'y'))
          {
                exit(0);
          }
      }

      printf("\n\nSETTING FORWARD AND REFLECTED PREAMP GAINS \n");
      printf("====================================================================\n\n");


      printf("Forward Gain = 26 \n");
	  printf("Reflected Gain = 22 \n");

	  for (i = 0; i < numOfDevices; i++)
	  {
		for (j = 0; j < 4; j++)
		{
			fwdGain[(i * 4) + j] = 26;
			reflGain[(i * 4) + j] = 22;
		}
	  }

      for (i = 0; i < numOfDevices; i++)
      {
            socket = socketsArranged[i];

            for (j = 0; j < 4; j++)
            {
                fGain = fwdGain[(i * 4) + j];
                rGain = reflGain[(i * 4) + j];

                rStatus = EthernetConfigurePreampGains(socket, j + 1, rGain, fGain);
            }
      }

      if( access( "tempExtClk.txt", R_OK|W_OK ) != -1 )
      {
          extClkConfigured = true;
          remove("tempExtClk.txt");
      }

      if( access( "tempLoCal.txt", R_OK|W_OK ) != -1 )
      {
          loCableCal = true;
          remove("tempLoCal.txt");
      }

      if( access( "tempThruCal.txt", R_OK|W_OK ) != -1 )
      {
          thruCal = true;
          remove("tempThruCal.txt");
      }

      if( access( "tempUnkThruCal.txt", R_OK|W_OK ) != -1 )
      {
          unkThruCal = true;
          remove("tempUnkThruCal.txt");
      }


      // write unit configuration setup to file (config.txt)
      fpw = fopen("config.txt", "w");

      fprintf(fpw, "Number of units: %d\n", numOfDevices);

      for (i = 0; i < numOfDevices; i++)
      {
          for (j = 0; j < 8; j++)
          {
              fprintf(fpw, "%c", serialNumbers[(i * 10) + j]);
          }

          fprintf(fpw, ": %d\n", unitNumber[i]);
      }

      fprintf(fpw, "\nFACTORY MULTI CAL:\n");
      fprintf(fpw, "%s\n", multiFactoryCal ? "True" : "False");
      fprintf(fpw, "\nUSER MULTI CAL:\n");
      fprintf(fpw, "%s\n", multiUserCal ? "True" : "False");
      fprintf(fpw, "\nCONFIGURED EXT CLOCK:\n");
      fprintf(fpw, "%s\n", extClkConfigured ? "True" : "False");
      fprintf(fpw, "\nLO CABLE CAL:\n");
      fprintf(fpw, "%s\n", loCableCal ? "True" : "False");
      fprintf(fpw, "\nTHROUGH CAL:\n");
      fprintf(fpw, "%s\n", thruCal ? "True" : "False");
      fprintf(fpw, "\nUNKNOWN THROUGH CAL:\n");
      fprintf(fpw, "%s\n", unkThruCal ? "True" : "False");

      fclose(fpw);

      // append gain settings and default sweep setup parameters to file (config.txt)
      fpw = fopen("config.txt", "a");

      fprintf(fpw, "\n\nMOST RECENT GAIN SETTINGS\n");
      fprintf(fpw, "=========================\n");

        for (i = 0; i < numOfDevices; i++)
        {
            for (j = 0; j < 4; j++)
            {
                fGain = fwdGain[(i * 4) + j];
                fprintf(fpw, "%d", fGain);

                if ((i < numOfDevices - 1) || (j < 3))
                {
                    fprintf(fpw, ",");
                }
            }
        }

        fprintf(fpw, "\n");

        for (i = 0; i < numOfDevices; i++)
        {
            for (j = 0; j < 4; j++)
            {
                rGain = reflGain[(i * 4) + j];
                fprintf(fpw, "%d", rGain);

                if ((i < numOfDevices - 1) || (j < 3))
                {
                    fprintf(fpw, ",");
                }
            }
        }

      fprintf(fpw, "\n\nMOST RECENT SWEEP SETUP\n");
      fprintf(fpw, "=======================\n");

          fprintf(fpw, "0\n");
          fprintf(fpw, "500000000\n");
          fprintf(fpw, "2000000000\n");
          fprintf(fpw, "4000000\n");
          fprintf(fpw, "1\n");
          fprintf(fpw, "32\n");
          fprintf(fpw, "0\n");
          fprintf(fpw, "1000\n");
          fprintf(fpw, "2\n");
          fprintf(fpw, "0\n\n");

      fclose(fpw);
      fflush(stdout);
  }

  // display device information of each connected unit

    int currUnitFwVer[4];
    memset(currUnitFwVer, 0, sizeof(currUnitFwVer));
    int t = 0;


  printf("\nCURRENT SETUP CONFIGURATION\n");
  printf("=========================\n\n");

  for (i = 0; i < numOfDevices; i++)
  {
        printf("[INFO] Socket : %d\n", socketsArranged[i]);
        printf("[INFO] Board Number : %d\n", unitNumber[i]);

  		rStatus = EthernetGetDeviceInformation(socketsArranged[i], &unitSerialNumber[0], (char*)&unitFirmwareVer[0], &unitTemperature);

  		if (unitNumber[i] == 1)
        {
            primarySocket = socketsArranged[i];

            for(loopcount = 0; loopcount < 10; loopcount++)
            {
                primarySerialNumber[loopcount] = unitSerialNumber[loopcount];
            }
        }

  		printf("[INFO] Serial Number : ");
  		for(loopcount = 0; loopcount < 10; loopcount++)
  		{
  			printf("%c", unitSerialNumber[loopcount]);
  		}
  		printf("\n");

  		t = 0;

  		printf("[INFO] Firmware Version : ");
		for(loopcount = 0; loopcount < 8; loopcount++)
		{
		    if (loopcount >= 4)
            {
                printf("%d",unitFirmwareVer[loopcount]);

                currUnitFwVer[t] = (int)(unitFirmwareVer[loopcount]);

                if (loopcount < 7)
                {
                    printf(".");
                }

                t++;
            }
		}
		printf("\n");

		for (counter = 0; counter < 8; counter++)
          {
              allUnitFirmwareVer[i][counter] = unitFirmwareVer[counter];
          }

		printf("[INFO] Temperature : %f\n", unitTemperature);

		printf("[INFO] Forward Gain: ");

        boardNum = unitNumber[i];

        for (int fg = (boardNum - 1) * 4; fg < ((boardNum - 1) * 4) + 4; fg++)
        {
            printf("%d", fwdGain[fg]);

            if (fg < ((boardNum - 1) * 4) + 3)
            {
                printf(",");
            }
        }

        printf("\n[INFO] Reflected Gain: ");

        for (int rg = (boardNum - 1) * 4; rg < ((boardNum - 1) * 4) + 4; rg++)
        {
            printf("%d", reflGain[rg]);

            if (rg < ((boardNum - 1) * 4) + 3)
            {
                printf(",");
            }
        }

        printf("\n\n");
  }

  fwDiff = CompareAllUnitFrimwareVer(numOfDevices, allUnitFirmwareVer);

  if (fwDiff)
  {
      printf("WARNING! Inconsistent firmware version detected among the connected units...\n");
      printf("WARNING! Application may not perform as intended in this condition...\n");
      printf("Please make sure all units are using the same firmware versions.\n");
	 
	printf("Do you want to continue despite the difference [Y/N] ");
	while ((verdict = GetInput(&input)) != 0)
	{
		printf("Please input 'Y' or 'y' to continue despite the error\n");
		printf("or input 'N' or 'n' to exit: ");
	}

	if (input == 'N' || input == 'n')
	{
		//rStatus = EthernetAbortOperation(numOfDevices, socketsArranged);
		rStatus = DisconnectEthernet(numOfDevices, socketsArranged);
		exit(EXIT_FAILURE);
	}
    
      
  }

    printf("\n\n!!!! REMINDER ON LO AND REFERENCE CLOCK SETTING !!!!\n");
    printf("====================================================\n");
    printf("[NOTE] Primary Unit Serial Number : ");

    for(loopcount = 0; loopcount < 10; loopcount++)
    {
        printf("%c", primarySerialNumber[loopcount]);
    }
    printf("\n");

    printf("[NOTE] Primary Unit Ethernet Socket : %d", primarySocket);
    printf("\n");
    printf("[IMPORTANT] The LO source and Reference Clock source will originate from this unit\n");
    printf("[IMPORTANT] Please make sure the connections are made appropriately on the backplane\n");
    printf("[IMPORTANT] >> Connection of [LO Out] and [Ref Out] starts from this unit <<\n");
    printf("[IMPORTANT] >> into [LO In] and [Ref In] of subsequent units in a daisy chain manner <<\n");
    printf("\n");



    // ********************** CAL SECTION *****************************************

    // This section manages the calibration factors or the combination calibration factors to be enabled

    // ---- Re-confirmation of FACTORY CAL DOWNLOAD
    if (!multiFactoryCal)
    {
      printf("\nMulti-unit FACTORY calibration factor table for these units has not been constructed yet... ");
      printf("\nDo you want to construct the multi-unit FACTORY calibration factor table for these units now [Y/N]? ");

      while ((verdict = GetInput(&input)) != 0)
      {
          printf("Please input 'Y' or 'y' to construct the multi-unit FACTORY calibration factor table\n");
          printf("or input 'N' or 'n' to skip");
      }

      if ((input == 'Y') || (input == 'y'))
      {
          // ******* Build Multi-unit Factory Cal Factors Table ******
          rStatus = EthernetDownloadCalCoeffAll(numOfDevices, deviceSocket);
          GetInputDataCorr(rStatus);
          ConstructMultiUnitFactoryCalTable(numOfDevices);
          rStatus = PopulateMultiUnitFactoryCalFactStruct(numOfDevices, testFactoryMultiCalFact);
          if (rStatus != 0)
            {
                exit(0);
            }
          multiFactoryCal = true;
      }
    }
    // ---- END: Re-confirmation of FACTORY CAL DOWNLOAD

    printf("\nDo you want to enable multi-unit calibration factors to all units [Y/N]? ");

    while ((verdict = GetInput(&input)) != 0)
    {
        printf("Please input 'Y' or 'y' to enable multi-unit calibration factors to all units\n");
        printf("or input 'N' or 'n' to disable multi-unit calibration factors to all units: ");
    }

    if ((input == 'Y') || (input == 'y'))
    {
        if (multiFactoryCal && multiUserCal)
        {
            printf("\nPlease input '1' for multi-unit FACTORY calibration factors\n");
            printf("or '2' for multi-unit USER calibration factors: ");

            while ((verdict = GetCalInput(&calInput)) != 0)
            {
                if (multiFactoryCal && multiUserCal)
                {
                    printf("\nPlease input '1' for multi-unit FACTORY calibration factors\n");
                    printf("or '2' for multi-unit USER calibration factors: ");
                }
            }
        }
        else if (multiFactoryCal && !multiUserCal)
        {
            printf("\nOnly Multi-unit FACTORY calibration factors table available. This will be enabled.\n");
            calInput = 1;
        }
        else if (!multiFactoryCal && multiUserCal)
        {
            printf("\nOnly Multi-unit USER calibration factors table available. This will be enabled.\n");
            calInput = 2;
        }
        else
        {
            printf("\nNo calibration factors table detected.\n");
            printf("\nDo you want to construct construct the multi-unit FACTORY calibration factors table now and enable it? [Y/N]\n");

            while ((verdict = GetInput(&input)) != 0)
            {
              printf("Please input 'Y' or 'y' to construct the multi-unit FACTORY calibration factor table\n");
              printf("or input 'N' or 'n' to skip");
            }

            if ((input == 'Y') || (input == 'y'))
            {
              // ******* Build Multi-unit Factory Cal Factors Table ******
              rStatus = EthernetDownloadCalCoeffAll(numOfDevices, deviceSocket);
              GetInputDataCorr(rStatus);
              ConstructMultiUnitFactoryCalTable(numOfDevices);
              rStatus = PopulateMultiUnitFactoryCalFactStruct(numOfDevices, testFactoryMultiCalFact);
              if (rStatus != 0)
                {
                    exit(0);
                }
              multiFactoryCal = true;
              calInput = 1;
            }
        }

        if (calInput != 0 && (thruCal || unkThruCal))
        {
            printf("\nDo you want to enable multi-unit THROUGH calibration factors to all units as well [Y/N]? ");

            while ((verdict = GetInput(&input)) != 0)
            {
                printf("Please input 'Y' or 'y' to enable multi-unit THROUGH calibration factors to all units\n");
                printf("or input 'N' or 'n' to skip: ");
            }

            if ((input == 'Y') || (input == 'y'))
            {
                if (thruCal && unkThruCal)
                {
                    printf("\nPlease input '1' for NORMAL THROUGH calibration factors\n");
                    printf("or '2' UNKNOWN THROUGH calibration factors: ");

                    while ((verdict = GetCalInput(&thruCalInput)) != 0)
                    {
                        if (multiFactoryCal && multiUserCal)
                        {
                            printf("\nPlease input '1' for NORMAL THROUGH calibration factors\n");
                            printf("or '2' UNKNOWN THROUGH calibration factors: ");
                        }
                    }
                }
                else if (thruCal && !unkThruCal)
                {
                    printf("\nOnly NORMAL THROUGH calibration factors table available. This will be enabled.\n");
                    thruCalInput = 1;
                }
                else if (!thruCal && unkThruCal)
                {
                    printf("\nOnly UNKNOWN THROUGH calibration factors table available. This will be enabled.\n");
                    thruCalInput = 2;
                }
            }

        }
		

        if (calInput == 1)
        {
            rStatus = PopulateMultiUnitFactoryCalFactStruct(numOfDevices, testFactoryMultiCalFact);
            if (rStatus != 0)
            {
                exit(0);
            }
            EthernetEnableMultiFactCalCoeffAll(true);
            EthernetEnableMultiUserCalCoeffAll(false);
        }
        else if (calInput == 2)
        {
            rStatus = PopulateMultiUnitUserCalFactStruct(numOfDevices, testUserMultiCalFact);
            if (rStatus != 0)
            {
                exit(0);
            }

            EthernetEnableMultiFactCalCoeffAll(false);
            EthernetEnableMultiUserCalCoeffAll(true);
        }
  // *********************************************************  New New New *****************************************************************************************      
  //  UknownAlgorithm = 0;                            // Set to 0 for old unknown Thru Algorithm *** Do not set to 1 unless given specific instructions to do so!! 
  //                                                  // Setting this to a 1 will force the New Unknown Thru Coefficients on which may not be correct!
  //	SelectCalibrationAlgorithm(UknownAlgorithm);  //  
        if (thruCalInput == 1)
        {
            rStatus = PopulateThruCalFactStruct(numOfDevices, "", thruCalInput, testThruCalFact);
            if (rStatus != 0)
            {
                exit(0);
            }
            EthernetEnableThruCalCoeffAll(true, thruCalInput);
        }
        else if (thruCalInput == 2)
        {   
	        rStatus = PopulateThruCalFactStruct(numOfDevices, "", thruCalInput, testThruCalFact);
            if (rStatus != 0)
            {
                exit(0);
            }
            EthernetEnableThruCalCoeffAll(true, thruCalInput);
        }
        else
        {
            EthernetEnableThruCalCoeffAll(false, thruCalInput);
        }

    }

        // -------------- S2P Configuration --------------

    int freqPoints = 0; // to store total number of frequency points in the s2p file
    bool saveRequired;
    bool runS2p = true;
	if ( (thruCalInput == 2) || (thruCalInput == 1))
	{		
		if (access("S2pCables.conf", R_OK) == -1)
		{
			printf("\nNo port extension configuration detected. Do you want to configure it now [Y/N]? ");

			while ((verdict = GetInput(&input)) != 0)
			{
				printf("Please input 'Y' or 'y' to configure port extension\n");
				printf("or input 'N' or 'n' to skip: ");
			}

			if ((input == 'Y') || (input == 'y'))
			{
				runS2p = true;
			}
			else

			{
				runS2p = false;
			}
		}
		else
		{
			printf("\n[Port extension configuration detected]...\n ");
			rStatus = CreateTMatrixStrucArrayFromCableS2pConfigFile("S2pCables.conf", &freqPoints);
		}


		if (runS2p)
		{
			ConfigureCable(&saveRequired);

			if (saveRequired)
			{
				SaveCableConfig("S2pCables.conf");
				rStatus = CreateTMatrixStrucArrayFromCableS2pConfigFile("S2pCables.conf", &freqPoints);
				if (rStatus == MN7021aERR_NONE)
				{
					ExportTMatrixValueToFile("TMatrixValue.csv", freqPoints);
				}
			}

			printf("\nDo you want to load and enable the current port extension correction factors [Y/N]? ");

			while ((verdict = GetInput(&input)) != 0)
			{
				printf("Please input 'Y' or 'y' to load and enable\n");
				printf("or input 'N' or 'n' to skip: ");
			}

			if ((input == 'Y') || (input == 'y'))
			{
				rStatus = CreateTMatrixStrucArrayFromCableS2pConfigFile("S2pCables.conf", &freqPoints);
				rStatus = LoadTMatrixValuesFromFile("TMatrixValue.csv");
				EthernetEnablePortExtCorrFact(true);
			}
			else
			{
				EthernetEnablePortExtCorrFact(false);
			}
		}
	}
	else
		EthernetEnablePortExtCorrFact(false);
        // -------------- S2P Configuration (END)--------------
 // ********************** ISOLATION CALIBRATION TERM SECTION ******************
    if(access("IsolationTerms.csv",R_OK) == 0)
    {
        printf("\nIsolation calibration term file detected...\n");
        printf("Do you want to use the isolation calibration term file [Y/N]? ");

        while ((verdict = GetInput(&input)) != 0)
        {
            printf("Please input 'Y' or 'y' to load and enable\n");
            printf("or input 'N' or 'n' to skip: ");
        }

        if ((input == 'Y') || (input == 'y'))
        {
			if(LoadIsolationTermValuesFromFile("IsolationTerms.csv",numOfDevices) != MN7021aERR_NONE)
            {
                printf("Isolation calibration term file read failed. Isolation term would not be used in calibration.\n");
            }
        }
    }
    // ********************** ISOLATION CALIBRATION TERM SECTION END **************
    // ********************** CAL SECTION *****************************************

    if (!extClkConfigured)
    {
        printf("[NOTE] Multi-unit external clock configuration has not been performed\n");
        printf("Configuring external clock now ... \n");

        for (i = 0; i < numOfDevices; i++)
        {
            if (unitNumber[i] == 1)
            {
                rStatus = EthernetConfigureExtClk(socketsArranged[i], 3);
                printf("[CONFIG] Unit 1 External clock config : Inverted\n");
            }
            else if (unitNumber[i] == numOfDevices)
            {
                rStatus = EthernetConfigureExtClk(socketsArranged[i], 2);
                printf("[CONFIG] Unit %d External clock config : Last\n", unitNumber[i]);
            }
            else
            {
                rStatus = EthernetConfigureExtClk(socketsArranged[i], 1);
                printf("[CONFIG] Unit %d External clock config : Normal\n", unitNumber[i]);
            }
            rStatus = EthernetRebootUnit(socketsArranged[i]);
            printf("###################################################################\n");
        }

        if (!loCableCal)
        {
            printf("\n\n************************ REMINDER ************************.\n");
            printf("LO cable calibration has not been performed yet.\nPlease run LO cable calibration after the units have been rebooted.\n\n");
        }


        printf("All MN7021A will auto reboot.\nPress ENTER to exit application... \n");


        fpr = fopen("config.txt", "r");
        fpw2 = fopen("temp.txt","w");
        if (fpr == NULL)
            exit(EXIT_FAILURE);


        while ((read = getline(&line, &len, fpr)) != -1)
        {
            if (strcmp(line, "FACTORY MULTI CAL:\n") == 0)
            {
                if (multiFactoryCal)
                {
                    fprintf(fpw2, "%s", line);
                    fprintf(fpw2, "True\n");
                    read = getline(&line, &len, fpr);
                    read = getline(&line, &len, fpr);
                }
            }

            if (strcmp(line, "CONFIGURED EXT CLOCK:\n") == 0)
            {
                fprintf(fpw2, "%s", line);
                fprintf(fpw2, "True\n");
                read = getline(&line, &len, fpr);
                read = getline(&line, &len, fpr);
                extClkWritten = true;
            }

            if (strcmp(line, "LO CABLE CAL:\n") == 0)
            {
                fprintf(fpw2, "%s", line);
                fprintf(fpw2, "False\n");
                read = getline(&line, &len, fpr);
                read = getline(&line, &len, fpr);
                loCalWritten = true;
            }

            if (strcmp(line, "THROUGH CAL:\n") == 0)
            {
                fprintf(fpw2, "%s", line);
                fprintf(fpw2, "False\n");
                read = getline(&line, &len, fpr);
                read = getline(&line, &len, fpr);
                thruCalWritten = true;
            }

            //if (strcmp(line, "MOST RECENT GAIN SETTINGS\n") == 0)
            //{
            //    if (!extClkWritten || !loCalWritten || !thruCalWritten)
            //    {
            //        if (!extClkWritten)
            //        {
            //            fprintf(fpw2, "CONFIGURED EXT CLOCK:\n");
            //            fprintf(fpw2, "True\n");
            //            fprintf(fpw2, "%s", line);
            //        }

            //        if (!loCalWritten)
            //        {
            //            fprintf(fpw2, "LO CABLE CAL:\n");
            //            fprintf(fpw2, "False\n");
            //            fprintf(fpw2, "%s", line);
            //        }
            //    }
            //    else
            //    {
            //        fprintf(fpw2, "%s", line);
            //    }
            //}
            else
            {
                fprintf(fpw2, "%s", line);
            }
        }

        fclose(fpr);
        fclose(fpw2);
        if (line)
        {
            free(line);
        }

        remove("config.txt");
        rename("temp.txt", "config.txt");

        WaitEnterKey();

	exit(0);
    }

    printf("Do you want to use the power calibration tables [Y/N]? ");

    while ((verdict = GetInput(&input)) != 0)
    {
        printf("Please input 'Y' or 'y' to load tables\n");
        printf("or input 'N' or 'n' to skip ");
    }
    if ((input == 'Y') || (input == 'y'))
    {
		printf("\nDo you want to reconstruct the Power Calibration Tables for these units [Y/N]? ");
		while ((verdict = GetInput(&input)) != 0)
		{
			printf("Please input 'Y' or 'y' to construct the Power Calibration Tables for these units\n");
			printf("or input 'N' or 'n' to skip");
		}
		if ((input == 'Y') || (input == 'y'))
		{
			  // ******* Build Multi-unit Power Factors Table ******
			rStatus = EthernetGetAllUnitsPowerCal(numOfDevices, deviceSocket);
		}
		rStatus = PopulatedPowerCalToStructArray(numOfDevices);
		if (rStatus == MN7021aERR_FILE_NOT_FOUND)
        {
            rStatus = EthernetGetAllUnitsPowerCal(numOfDevices, deviceSocket);
            rStatus = PopulatedPowerCalToStructArray(numOfDevices);
        }
		if (rStatus == MN7021aERR_NONE)
		{
			printf("Power Tables Loaded\nPlease Enter Power Level (in dBm)? ");
			GetFloatInput(&powerLevel);
			printf("Enter frequency for the power level to be adjusted for? "); 
			GetDoubleInput(&powerFrequency);			
			rStatus = SetSystemPowerLevel(numOfDevices, socketsArranged, powerLevel, powerFrequency,  powerPortSet);
            if (rStatus != MN7021aERR_NONE)
            {
                printf("Error Code %d = %s\nPower not set.\n",rStatus, MN7021aErrToString(rStatus)); // print string error code
            }
            else
            {
                for (i = 0;i < numOfDevices*4 ;i++)
                {
                    printf(" Port %d Setting %d Power %f ",i+1, powerPortSet[i], CalcPowerFromStructArray(numOfDevices,  powerPortSet[i], powerFrequency, i+1));
                    printf(" Power at start frequency %f Power at stop frequency %f\n", CalcPowerFromStructArray(numOfDevices,  powerPortSet[i], startF, i+1), CalcPowerFromStructArray(numOfDevices,  powerPortSet[i], stopF, i+1));
                }
            }
		}
        else
        {
            printf("Error Code %d = %s\nPower tables not loaded.\n",rStatus, MN7021aErrToString(rStatus)); // print string error code
        }	
	}

    if (!loCableCal && (numOfDevices > 1))
    {
        printf("\n\n************************ REMINDER ************************.\n");
        printf("LO cable calibration has not been performed yet.\nIt is highly recommended to run LO cable calibration before performing any sweep measurements.\n");

        printf("Do you want to exit the application and run LO cable calibration first [Y/N]? ");

        while ((verdict = GetInput(&input)) != 0)
        {
            printf("Please input 'Y' or 'y' to exit the application\n");
            printf("or input 'N' or 'n' to continue without LO cable cal: ");
        }

        // user input to re-setup sweep parameters
        if ((input == 'Y') || (input == 'y'))
        {
            exit(0);
        }
    }

    // Check secondary units LO Detection and Lock status

    printf("\n###################################################################\n");

    printf("REF CLOCK DETECTION, REFERENCE LOCK DETECTION and HEALTH STATUS\n");

    allLocked = true;

    for (i = 0; i < numOfDevices; i++)
    {
        rStatus = EthernetQuerySystemStatus(socketsArranged[i], &systemStat);
        if ( systemStat != 0 )
        {
            printf("**** Warning Status flag set on Unit %d. Status code = %0x  \n",unitNumber[i],systemStat);

            allLocked = false;

            if ((systemStat & OverTempMask) != 0)
            {
               rStatus = EthernetGetDeviceInformation(socketsArranged[i], &unitSerialNumber[0], (char*)&unitFirmwareVer[0], &unitTemperature);
               printf("Over Temp detected on unit # %d  Temperature = %f\n", i + 1, unitTemperature );
            }
        if ((systemStat & CriticalTempMask) != 0)
            {
               rStatus = EthernetGetDeviceInformation(socketsArranged[i], &unitSerialNumber[0], (char*)&unitFirmwareVer[0], &unitTemperature);
               printf("[ERROR] Critical Temp detected on unit # %d  Temperature = %f\n", i + 1, unitTemperature );
            }
			
		if ((systemStat & FanRPMFaultMask) != 0)
            {
               rStatus = EthernetGetDeviceInformation(socketsArranged[i], &unitSerialNumber[0], (char*)&unitFirmwareVer[0], &unitTemperature);
               printf("[ERROR] Fan RPM Fault Detected\n");
            }
		if ((systemStat & FanOverCurrentMask) != 0)
            {
               rStatus = EthernetGetDeviceInformation(socketsArranged[i], &unitSerialNumber[0], (char*)&unitFirmwareVer[0], &unitTemperature);
               printf("[ERROR] Fan Over Current Detected\n" );
            }
            if ((systemStat & ADCCLKPLLMask) != 0)
            {
               printf("ADC Clock PLL Problem detected on unit # %d \n", i + 1);
            }

            if ((systemStat & RTCBattMask) != 0)
            {
               printf("RTC Battery Low detected on unit # %d \n", i + 1);
            }

            if ((systemStat & RefUnlockMask) != 0)
            {
               printf("UNIT %d: 100MHz REF external clock not detected\n", i + 1);
            }

            if ((systemStat & RefInputLowSigMask) != 0)
            {
               printf("UNIT %d: External Reference Input Amplitude Low\n", i + 1);
            }
       }

  rStatus = getUnitPersistFlags(socketsArranged[i], &persistFlags);
  if (rStatus == MN7021aERR_INSTRUMENT_STATE)
	{
		printf("[WARNING] Unit Persistence Flag is set: \n");
		if (persistFlags.criticalTemp)
			printf("- Over Critical Temp Event \n");
		if (persistFlags.fanRpmFault)
			printf("- Fan RPM Fault Event \n");
		if (persistFlags.fanOverCurrent)
			printf("- Fan Over Current Event");
		printf("\n");
		if (persistFlags.fanInstall)
			printf("[INFO] Fan RPM Check Enabled\n");
		else
			printf("[INFO] Fan RPM Check disabled\n");
	 }
     if (rStatus == MN7021aERR_NONE)
     {	
		if (persistFlags.fanInstall)
			printf("[INFO] Fan RPM Check Enabled\n");
		else
			printf("[INFO] Fan RPM Check disabled\n");
	}

    printf("###################################################################\n");
    }
    if (!allLocked)
    {
        printf("WARNING!!! Problem(s) detected\n");
        printf("WARNING!!! Sweep results may not be correct\n");
        printf("Problem detected with one or more units do you want to continue [Y/N]? ");

        while ((verdict = GetInput(&input)) != 0)
        {
            printf("Please input 'Y' or 'y' to continue \n");
            printf("or input 'N' or 'n' to end program ");
        }
            // user input to re-setup sweep parameters
        if ((input == 'N') || (input == 'n'))
        {
            printf("Closing Sockets and exiting program\n");
            rStatus = EthernetAbortOperation(numOfDevices, deviceSocket);
            rStatus = DisconnectEthernet(numOfDevices, deviceSocket);
            exit(-1);
        }
    }
    else{
        printf("All %d unit(s) Health and Status checks are correct\n\n",numOfDevices);
    }

    if (!allLocked)
    {
        printf("WARNING!!! Not all secondary units had detected/locked on to the 100MHz REF external clock\n");
        printf("WARNING!!! Sweep results may not be correct\n\n");
    }
    systemInit = GetSystemInitState(numOfDevices, &unitNumber[0], socketsArranged);

    if (systemInit)
    {
		printf("System is initialized and ADC Clocks aligned\n\n");
	}
	else
		printf("System is not initialized\n\n");

    if ((numOfDevices > 1) && loCableCal)
    {
        rStatus = CheckAndReadLOFile(numOfDevices, socketsArranged, delayTable);
    }

    // display default/most recent sweep setup parameters
    printf("\n\nMOST RECENT SWEEP SETUP\n");
    printf("===================\n");

    if (swpType == 0)
    {
        printf("Sweep Type: Frequency Sweep\n");
    }
    else
    {
        printf("Sweep Type: Port Sweep\n");
    }

    if(segSwp == 0)
    {
        printf("Start Frequency: %9.0f Hz\n", startF);
        printf("Stop Frequency: %9.0f Hz\n", stopF);
        printf("Frequency Step: %9.0f Hz\n", stepF);
        printf("Sweep count: %d\n", swpCnt);
        printf("IF Bandwidth: %s\n", GetIFBWEnumStringFromUInt16(ifBw));
        printf("Segmented Sweep Disabled\n");
        
    }
    else 
    {
        printf("Sweep count: %d\n", swpCnt);
        printf("Segmented Sweep Enabled\n");

    }
    
    if (mode == MODE1)
    {
        printf("Sweep mode: MODE1\n");
    }
    else
    {
        printf("Sweep mode: MODE0\n");
    }

    printf("Delay between sweep(us): %d (only applied when sweep mode is set to MODE1)\n", delayBtwSwp);

    if (saveMode == SAVETOMEM)
    {
        printf("Save results: to mapped memory\n");
    }
    else if (saveMode == SAVETOFILE)
    {
        printf("Save results: to CSV file\n");
    }
    else if (saveMode == SAVETOMEMANDFILE)
    {
        printf("Save results: to mapped memory and CSV file\n");
    }
    else
    {
        printf("Save results: none\n\n");
    }

    printf("Do you want to perform sweep with the default setup [Y/N]? ");

    while ((verdict = GetInput(&input)) != 0)
    {
        printf("Please input 'Y' or 'y' to continue with this configuration\n");
        printf("or input 'N' or 'n' for a different configuration: ");
    }

    // user input to re-setup sweep parameters
    if ((input == 'N') || (input == 'n'))
    {
        reOptimize = true;

    
        swpType = 0;
		
        printf("Please choose Sweep State [0-Normal Sweep / 1-Segmented Sweep]: ");
        GetIntegerInputWithRange(&segSwp, 0, 1, true, true);

        if(segSwp == 0)
        {
            printf("Please input Start Frequency: ");
            GetDoubleInput(&startF);
            printf("Please input Stop Frequency: ");
            GetDoubleInput(&stopF);
            printf("Please input Frequency Step: ");
            GetDoubleInput(&stepF);
            CheckSweepPoints(&startF, &stopF, &stepF);
			printf("Number of sweeps within the block [1 - %d]: ",MAX_SWEEP_COUNT);
			GetIntegerInputWithRange(&swpCnt, 1, MAX_SWEEP_COUNT, true, true);
            printf("Please input IF Bandwidth\n");
            for (j=0;j < (sizeof(ifTable) / sizeof(ifTable[0]));j++)
            {
                printf("   %d = %s\n",j,ifTable[j]);
            }
            printf("Enter IF Integer number: ");
            GetIntegerInput(&ifBwExp);
            if ((ifBwExp >= (sizeof(ifTable) / sizeof(ifTable[0]))) | (ifBwExp < 0))
            {
                printf("\n%d is out of range setting to default 390KHz\n",ifBwExp);
                ifBwExp = 4;
            }
            ifBw = ifSetting[ifBwExp];
            printf("\n");
            printf("IF BW = %s\n",ifTable[ifBwExp]);
			modeIndex = 0;
            mode = swMode[modeIndex];
            printf("\n");

            if (mode == MODE1)
            {
                printf("Please input delay between sweep in microseconds: ");
                GetIntegerInput(&delayBtwSwp);
            }
        }
        else 
        {
            printf("Number of sweeps within the block [1 - %d]: ",MAX_SWEEP_COUNT);
			GetIntegerInputWithRange(&swpCnt, 1, MAX_SWEEP_COUNT, true, true);
            mode = swMode[0];
        }


        printf("0 - SAVE TO NONE\n");
        printf("1 - SAVE TO MAPPED MEMORY\n");
        printf("2 - SAVE TO CSV FILE\n");
        printf("3 - SAVE TO MAPPED MEMORY AND CSV FILE\n");
        printf("Please choose result saving mode: ");
        GetIntegerInput(&saveModeIdx);
        if ((saveModeIdx < 0) || (saveModeIdx > 3))
        {
            printf("\n%d is out of range setting to default 2 - SAVE TO CSV FILE\n", saveModeIdx);
            saveModeIdx = 2;
        }
        saveMode = saveModeArr[saveModeIdx];
        printf("\n");

    	printf("Do you want to proceed to perform sweep [Y/N]? ");

        while ((verdict = GetInput(&input)) != 0)
        {
        printf("Please input 'Y' or 'y' to continue with this configuration\n");
        printf("or input 'N' or 'n' to exit ");
        }


        if ((input == 'N') || (input == 'n'))
        {
            exit(0);
        }
        }

        if (swpType == 1)
        {
            swpTypeInput = 1;
        }
        else
        {
            swpTypeInput = 0;
        }




    // re-write configuration file with user selected sweep setup parameters
    // NOTE: copy lines from config.txt into temp.txt until sweep parameters section.
    // Write new available sweep parameters into temp.txt
    // Remove config.txt (previous configuration) and rename temp.txt to config.txt

    fpr = fopen("config.txt", "r");
    fpw2 = fopen("temp.txt","w");
    if (fpr == NULL)
        exit(EXIT_FAILURE);


    lineCnt = 0;
    while ((read = getline(&line, &len, fpr)) != -1)
    {
        if (strcmp(line, "FACTORY MULTI CAL:\n") == 0)
        {
            if (multiFactoryCal)
            {
                fprintf(fpw2, "%s", line);
                fprintf(fpw2, "True\n");
                read = getline(&line, &len, fpr);
                read = getline(&line, &len, fpr);
            }
        }

        if (strcmp(line, "MOST RECENT SWEEP SETUP\n") == 0)
        {
            fprintf(fpw2, "%s", line);
            read = getline(&line, &len, fpr);
            fprintf(fpw2, "%s", line);
            readFreq = 1;
        }
        else if ((readFreq > 0) && (readFreq < 11))
        {
            switch (readFreq)
            {
                case 1:
                {
                    fprintf(fpw2, "%d\n", swpType);
                    break;
                }

                case 2:
                {
                    fprintf(fpw2, "%f\n", startF);
                    break;
                }

                case 3:
                {
                    fprintf(fpw2, "%f\n", stopF);
                    break;
                }

                case 4:
                {
                    fprintf(fpw2, "%f\n", stepF);
                    break;
                }

                case 5:
                {
                    fprintf(fpw2, "%d\n", swpCnt);
                    break;
                }

                case 6:
                {
                    fprintf(fpw2, "%d\n", ifBw);
                    break;
                }

                case 7:
                {
                    fprintf(fpw2, "%d\n", mode);
                    break;
                }

                case 8:
                {
                    fprintf(fpw2, "%d\n", delayBtwSwp);
                    break;
                }

                case 9:
                {
                    fprintf(fpw2, "%d\n", saveMode);
                    break;
                }

                case 10:
                {
                    fprintf(fpw2, "%d\n", segSwp);
                    break;
                }

                default:
                    break;
            }

            readFreq++;
        }
        else
        {
            fprintf(fpw2, "%s", line);
        }

        lineCnt++;
    }

    fclose(fpr);
    fclose(fpw2);
    if (line)
    {
        free(line);
    }

    remove("config.txt");
    rename("temp.txt", "config.txt");

    // --------------------- Set Result Format (Real-Imaginary / Magnitude-Phase) -------------------------
    resultFormat = MAG_PHASE;

    // BATCH SWEEP FIX: Force saveMode to include memory storage. batch_sweep
    // needs the shared memory buffers populated so it can read sweep results
    // and write its own per-trial labeled CSVs. Without this, if config.txt
    // has saveMode=SAVETOFILE (2), shared memory is never allocated below and
    // every batch CSV comes out filled with zeros.
    saveMode = SAVETOMEMANDFILE;

    if (saveMode == SAVETOMEM || saveMode == SAVETOMEMANDFILE)
    {
        // fd1 = shm_open(STORAGE_ID1, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        // if (fd1 == -1)
        // {
        //     perror("open");
        // }

        // fd2 = shm_open(STORAGE_ID2, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        // if (fd2 == -1)
        // {
        //     perror("open");
        // }

        int res;
        // int memorySize = MAX_SWEEP_COUNT * CHAN_DATA_LEN * sizeof(double);

        // res = ftruncate(fd1, memorySize);
        // if (res == -1)
        // {
        //     perror("ftruncate");
        //     return 2;
        // }

        // res = ftruncate(fd2, memorySize);
        // if (res == -1)
        // {
        //     perror("ftruncate");
        //     return 2;
        // }

        EthernetSetStorageID(STORAGE_ID1, STORAGE_ID2);

        res = OpenSharedMemory(STORAGE_ID1, STORAGE_ID2, &fd1, &fd2);
        if (res != 0)
        {
            printf("ERROR! Failed to open shared memory...\n");
        }


		//>>>>> This FILE pointer is part of test code used to demonstrate printing sweep results
		//>>>>> saved in shared memory into CSV file
        // testfPtr = fopen("./Data/SPARAM_ReArrTest.csv", "w");
    }

    rStatus = VerifyFrequencyParameters(startF, stopF, stepF);

    if (rStatus == MN7021aERR_PARAM)
    {
        printf("ERROR! Start, stop, or step frequency are invalid. Exiting test...\n");
        exit(-1);
    }
    


// *********************** CAL OPTIMIZATION ***********************

   //rewriteConfigFile = false;
   EthernetOptimizeCalCoeff(numOfDevices, fileAppliedCal, &startF, &stopF, &stepF, loCableCal, reOptimize, false);



    rStatus = EthernetEnableParallelSweep(numOfDevices, 0);
// *********************** SYSTEM INITIALIZATION ***********************

    systemInit = GetSystemInitState(numOfDevices, &unitNumber[0], socketsArranged);

    if (!systemInit)
    {
		 
        printf("System is not fully initialized yet. Do you want to perform cold system initialization [Y/N]?");

        while ((verdict = GetInput(&input)) != 0)
        {
            printf("Please input 'Y' or 'y' to perform cold system initialization\n");
            printf("or input 'N' or 'n' to for regular initialization ");
        }

        if ((input == 'Y') || (input == 'y'))
        {
			printf("Enter maximum number minutes for synth warm up time? ");
			scanf("%f",&MaxWarmTime);
			printf("Running Synth warm up, ADC initialization and Calibration\n");
			rStatus = ColdSystemInitialization(numOfDevices, &unitNumber[0], socketsArranged, true, MaxWarmTime, SweepCtrlType);
			printf("Completed Synth Warm up, and ADC initialization\n");
		}
		else
		{
			printf("Running ADC initialization and Calibration\n");
			rStatus = SystemInitialization(numOfDevices, &unitNumber[0], socketsArranged, false, MaxTempDelta, SweepCtrlType);
		}
		if (rStatus != MN7021aERR_NONE)
		{
			printf("Error Code %d = %s\nSystem initialization errors encountered.\n",rStatus, MN7021aErrToString(rStatus)); // print string error code
			printf("Do you want to continue [Y/N] ");
			while ((verdict = GetInput(&input)) != 0)
			{
				printf("Please input 'Y' or 'y' to continue despite the error\n");
				printf("or input 'N' or 'n' to exit: ");
			}

			if (input == 'N' || input == 'n')
			{
				rStatus = EthernetAbortOperation(numOfDevices, socketsArranged);
				rStatus = DisconnectEthernet(numOfDevices, socketsArranged);
				exit(-1);
			}
		}
		else
			printf("ADC Clock Initialization successful\n");
		ADCcalComplete = true;

		printf("Do you want to wait for thermal stabilization [Y/N]? ");

		while ((verdict = GetInput(&input)) != 0)
		{
			printf("Please input 'Y' or 'y' to wait for thermal stabilization\n");
			printf("or input 'N' or 'n' not to wait ");
		}

		if ((input == 'Y') || (input == 'y'))
		{
			warmupWait = true;
		}

		rStatus = SystemInitialization(numOfDevices, &unitNumber[0], socketsArranged, warmupWait, MaxTempDelta, SweepCtrlType);

		if (warmupWait && rStatus == MN7021aERR_NONE)
		{
			if (!TempStabilize_Instance.complete)
				printf("Waiting thermal stabilization complete...\n");
			
			while (!TempStabilize_Instance.complete)
			{
				sleep(31);

				for (int i = 0; i < numOfDevices; i++)
				{
					printf("Unit %d, CPU Temp = %4.1f, Power supply Temp = %4.1f, Pattern Gen Temp = %4.1f, Bridge Port 1 Temp = %4.1f, Bridge Port 2 Temp = %4.1f, Bridge Port 3 Temp = %4.1f, Bridge Port 4 Temp = %4.1f, Ch1 DeltaT = %f, Ch2 DeltaT = %f, Ch3 DeltaT = %f, Ch4 DeltaT = %f\n",
					i + 1, TempStabilize_Instance.unitTemp[i][0], TempStabilize_Instance.unitTemp[i][1], TempStabilize_Instance.unitTemp[i][3], TempStabilize_Instance.unitTemp[i][4], TempStabilize_Instance.unitTemp[i][5], TempStabilize_Instance.unitTemp[i][6], TempStabilize_Instance.unitTemp[i][2],
					TempStabilize_Instance.unitTemp[i][7], TempStabilize_Instance.unitTemp[i][8], TempStabilize_Instance.unitTemp[i][9], TempStabilize_Instance.unitTemp[i][10]);
				}
				printf("****** Total System Delta = %f ******\n", TempStabilize_Instance.systemDeltaT);
			}

			if (TempStabilize_Instance.rStatus != MN7021aERR_NONE)
			{
				printf("Temp Stab error %d\n",TempStabilize_Instance.rStatus);
				rStatus = MN7021aERR_SYSTEM_INIT_FAIL;
			}
			else
			{
				printf("Thermal stabilization Done...\n\n");
			}
		}

		if (rStatus != MN7021aERR_NONE)
		{
			printf("Error Code %d = %s\nSystem initialization errors encountered.\n",rStatus, MN7021aErrToString(rStatus)); // print string error code
			printf("Do you want to continue [Y/N] ");
			while ((verdict = GetInput(&input)) != 0)
			{
				printf("Please input 'Y' or 'y' to continue despite the error\n");
				printf("or input 'N' or 'n' to exit: ");
			}

			if (input == 'N' || input == 'n')
			{
				rStatus = EthernetAbortOperation(numOfDevices, socketsArranged);
				rStatus = DisconnectEthernet(numOfDevices, socketsArranged);
				exit(-1);
			}
		}

		systemInit = GetSystemInitState(numOfDevices, &unitNumber[0], socketsArranged);

		if (systemInit)
		{
			ADCcalComplete = true;
			printf("System is fully initialized. Proceeding to sweep...\n");
		}
		else
		{
			printf("WARNING! System initialization was NOT successful. Exiting test...\n");
			exit(0);
		}
	
        
    }
    else
    {
        printf("System is fully initialized.\n");
    }

    rStatus = getUnitSocketState(numOfDevices, &unitNumber[0], 1, 0);
    if (UnitSockState.systemSocketReadyState == true)
    {
        printf("[INFO] System unit's socket is in ready state\n");
    }
    else
    {
        printf("[WARNING] System unit's socket is not in ready state\n");
        exit(0);
    }
    
// *********************** SYSTEM INITIALIZATION (END) ***********************

    ///////// Demo system test //////
	// rStatus = SystemEcalTest(numOfDevices,  &unitNumber[0], socketsArranged, true);
    // printf("System Test result = %d  = %s \n",rStatus,MN7021aErrToString(rStatus));
	///////// Demo system test //////
    char healthStatRetMd = 0;
    int monitorItr = 2;
    monitorItr = monitorItr;
    char cusFilename[200];

    if (segSwp == 1)
    {   
        printf("\nDo you want to enable Port Segmentation Sweep [Y/N]? ");
        while ((verdict = GetInput(&input)) != 0)
        {
            printf("Please input 'Y' or 'y' to enable\n");
            printf("or input 'N' or 'n' to disable Port Segmentation Sweep: ");
        }
        
        if ((input == 'Y') || (input == 'y'))
        {
            segChannel = 1;
            printf("Sweep with Channel Selection is Enabled\n");
            printf("The number of unit connected: %d\n",numOfDevices);
            printf("Please enter the active port for the segmented sweep [Enter '1' to enable or '0' disable the port]: \n\n");
            for (i = 0; i < numOfDevices; i++)
            {
                segCh = 1;
                if (i == 0)
                {
                    segMin = 0;
                    segMax = 3;
                }
                else if (i == 1)
                {
                    segMin = 4;
                    segMax = 7;
                }
                else if (i == 2)
                {
                    segMin = 8;
                    segMax = 11;
                }
                else
                {
                    segMin = 12;
                    segMax = 15;
                }

                for (segMin; segMin <= segMax; segMin++)
                {
                    printf("Unit %d Channel %d: ", i+1,segCh);
                    segCh++;
                    GetIntegerInputWithRange(&segPort[segMin], 0, 1, true, true);
                    if (segPort[segMin] == 1) 
                    {
                        segActivePort |= (segCheck << segMin); // set the corresponding bit to 1 using bitwise OR
                    }   
                }

            }

        }

        else
        {
            for (int k = 0; k < 16; k++)
            {
                segPort[k] = 1;
            }
            segChannel = 0;
        }

        rStatus = EthernetEnableSegmentedSweep(numOfDevices, segSwp);
        if (rStatus == MN7021aERR_LIC_FEATURE_NOT_INSTALLED || rStatus == MN7021aERR_NOT_IMPLEMENTED)
        {
            exit(0);
        }
        rStatus = SourcePortSelection(numOfDevices, segActivePort, segChannel);
        if (rStatus == MN7021aERR_LIC_FEATURE_NOT_INSTALLED)
        {
            exit(0);
        }
        if (segChannel == 1)
        {
            printf("\nDo you want to enable the high gain mode on non-active transmit ports [Y/N]? ");
            while ((verdict = GetInput(&input)) != 0)
            {
                printf("Please input 'Y' or 'y' to enable\n");
                printf("or input 'N' or 'n' to disable the high gain mode: ");
            }
            
            if ((input == 'Y') || (input == 'y'))
                highGainState = 1;
            else
                highGainState = 0;
            
            rStatus = HighGainSegPorts(numOfDevices, highGainState);
            if (rStatus == MN7021aERR_LIC_FEATURE_NOT_INSTALLED)
            {
                exit(0);
            }
        }

        printf("Do you want to load the the segmented sweep configuration file from different path [Y/N]?");
    
        while ((verdict = GetInput(&input)) != 0)
        {
            printf("Please input 'Y' or 'y' to load\n");
            printf("or input 'N' or 'n' to skip the load: ");
        }
        
        if (input == 'Y' || input == 'y')
        {
            printf("Enter the path and filename(.csv): "); 
            scanf(" %s", CustomSegmentConfigFile);
            printf("\n");

            while (access(CustomSegmentConfigFile, R_OK) == -1)
            {
                printf("Invalid filepath, please re-enter the file path and name : "); 
                scanf(" %s", CustomSegmentConfigFile);
                printf("\n");
            }

            rStatus = LoadSegmentedConfigFromFile(CustomSegmentConfigFile);
            if (rStatus == MN7021aERR_LIC_FEATURE_NOT_INSTALLED)
            {
                free(CustomSegmentConfigFile);
                exit(0);
            }

            free(CustomSegmentConfigFile);
        } 
        else 
        {
            printf("Do you want to generate the segmented sweep configuration file using a custom frequency set [Y/N]?: ");
            while ((verdict = GetInput(&input)) != 0)
            {
                printf("Please input 'Y' or 'y' to use custom\n");
                printf("or input 'N' or 'n' to use default: ");
            }

            if (input == 'Y' || input == 'y')
            {   
                printf("Please input Start Frequency: ");
                GetDoubleInput(&startF);
                printf("Please input Stop Frequency: ");
                GetDoubleInput(&stopF);
                printf("Please input Frequency Step: ");
                GetDoubleInput(&stepF);
                CheckSweepPoints(&startF, &stopF, &stepF);
                     
                printf("Enter the path and filename(.csv): "); 
                scanf(" %s", CustomSegmentConfigFile);
                printf("\n");

                constructSegmentConfigFile(CustomSegmentConfigFile, startF, stopF, stepF, ifBw, maxPower);
                printf("[INFO] Custom CSV file generated: %s\n", CustomSegmentConfigFile);

                rStatus = LoadSegmentedConfigFromFile(CustomSegmentConfigFile);
                if (rStatus == MN7021aERR_LIC_FEATURE_NOT_INSTALLED)
                {
                    free(CustomSegmentConfigFile);
                    exit(0);
                }

                free(CustomSegmentConfigFile);
                
            }
            else
            {
                if (access(SegmentConfigFile, R_OK|W_OK) == -1)
                {

                    printf("[INFO] %s not found. Generating CSV using the default frequency set....\n", SegmentConfigFile);
                    constructSegmentConfigFile(SegmentConfigFile, startF, stopF, stepF, ifBw, maxPower);
                }

                rStatus = LoadSegmentedConfigFromFile(SegmentConfigFile);
                if (rStatus == MN7021aERR_LIC_FEATURE_NOT_INSTALLED)
                {
                    exit(0);
                }
            }
        }
        
        rStatus = EthernetOptimizeCalCoeff(numOfDevices, fileAppliedCal, &startF, &stopF, &stepF, loCableCal, reOptimize, false);
        if (rStatus != MN7021aERR_NONE)
        {
            printf("Error Code %d = %s.\n",rStatus, MN7021aErrToString(rStatus));
            printf("Do you want to continue despite the warning [Y/N] ");
            while ((verdict = GetInput(&input)) != 0)
            {
                printf("Please input 'Y' or 'y' to continue despite the warning\n");
                printf("or input 'N' or 'n' to exit: ");
            }

            if (input == 'N' || input == 'n')
			{
				// rStatus = EthernetAbortOperation(numOfDevices, socketsArranged);
				rStatus = DisconnectEthernet(numOfDevices, socketsArranged);
				exit(-1);
            }
        }
    }

    printf("\nDo you want average multiple block sweep data to one result [Y/N]? ");

    while ((verdict = GetInput(&input)) != 0)
    {
        printf("Please input 'Y' or 'y' to enable\n");
        printf("or input 'N' or 'n' to disable\n");
    }

    if (input == 'Y' || input == 'y')
    {
        avgSwpData = true;
    }
    else
    {
        avgSwpData = false;
    }
    EthernetEnableAveragingSweepData(avgSwpData);

    printf("\nDo you want save data to specific path and name [Y/N]? ");
    
    while ((verdict = GetInput(&input)) != 0)
    {
        printf("Please input 'Y' or 'y' to save\n");
        printf("or input 'N' or 'n' to skip and exit: ");
    }
    
    if (input == 'Y' || input == 'y')
    {
        printf("Please enter your file path and name : "); 
        scanf(" %s", cusFilename);
        printf("\n");

        while(!(fopen(cusFilename, "w")))
        {
            printf("Invalid filepath, please re-enter your file path and name : "); 
            scanf(" %s", cusFilename);
            printf("\n");
        }

        remove(cusFilename);

        setCustomFilenameArr(cusFilename);
    }
    else
    {
        //setCustomFilenameArr("");
    }

    printf("Do you want to calculate the active sweep time based on the configuration [Y/N]? ");
    
    while ((verdict = GetInput(&input)) != 0)
    {
        printf("Please input 'Y' or 'y' to calculate\n");
        printf("or input 'N' or 'n' to skip and exit: ");
    }
    
    if (input == 'Y' || input == 'y')
    {
        if (segSwp == 0)
            printf("Calulated Sweep Time = %f Seconds\n\n",CalculateSweepTime(numOfDevices, swpCnt, (int)(floor((stopF - startF) / stepF)) + 1, ifBw));
        else
            printf("Calulated Sweep Time = %f Seconds\n\n",CalculateSweepTimeSeg(numOfDevices, swpCnt, segNumOfFreqPoints, SegmentSweep));
    }


    if (swpCnt > 1)
    {
        printf("\nDo you want enable system health status supervision monitoring [Y/N]? ");

        while ((verdict = GetInput(&input)) != 0)
        {
            printf("Please input 'Y' or 'y' to save\n");
            printf("or input 'N' or 'n' to skip and exit: ");
        }

        if (input == 'Y' || input == 'y')
        {
            healthStatRetMd = 2;
            monitorItr = 2;
            ComputeRecSupvParam(swpCnt, &threadArgs.intervalus);

            printf("\nRECOMMENDED SUPERVISION MONITORING PARAMETERS:\n");
            printf("Health status return sweep iteration interval = 2\n");
            printf("Health status monitoring  interval (in us) = %d\n", threadArgs.intervalus);
        }
        else
        {
            healthStatRetMd = 0;
        }
    }
    else
    {
        healthStatRetMd = 0;
    }

    rStatus = EthernetSetHeatlhStatusReturnMode(numOfDevices, socketsArranged, healthStatRetMd);

    if (healthStatRetMd > 1)
    {
        rStatus = EthernetSetHeatlhStatusSweepIteration(numOfDevices, socketsArranged, 2, swpCnt);

        if (rStatus != MN7021aERR_NONE)
        {
            printf("Set health status sweep iteration failed.\n");
            DisconnectEthernet(numOfDevices, deviceSocket);
            exit(0);
        }
    }


# if 0 // EXAMPLE of QUERYING SYstem Health Status
    rStatus = EthernetQuerySystemHealthStatus(numOfDevices, socketsArranged);

    HealthStatus tryStat[4];
    ExtractSystemHealthStatus(numOfDevices, &tryStat[0]);
# endif



    //  0 = FrequencySweep; 1 = PortSweep
    // run sweep



    // pthread_t supThrId;
    pthread_t abortThrId;
    parent = pthread_self(); // get thread ID of current sweep thread

    if (segSwp == 1)
    {
        signal(SIGINT, sig_handler);

        

        set_conio_terminal_mode();
        pthread_create(&abortThrId, NULL, AbortSweepIntrpt, NULL); // create and run abort interrupt thread

        rStatus = EthernetInitSegmentedFrequencySweepRawData(numOfDevices, swpTypeInput, &unitNumber[0], socketsArranged, resultFormat, swpCnt, false, saveMode, segPort);
        
        if (rStatus == MN7021aERR_LIC_FEATURE_NOT_INSTALLED)
        {
            exit(0);
        }

        if (rStatus != MN7021aERR_NONE)
        {
            printf("Error during Frequency Sweep\n\n");
        }

        signal(SIGINT, SIG_DFL);

        complete = CheckSweepComplete();

        if (complete == 0)
        {
            complete = 1; //reset it
        }


        pthread_cancel(abortThrId); // cancel abort interrupt thread
        pthread_detach(abortThrId);

        reset_terminal_mode();

        if (rStatus == MN7021aERR_NONE)
        {
            if (saveMode == SAVETOMEM || saveMode == SAVETOMEMANDFILE)
            {
                ReadResultFromMemorySegmented(STORAGE_ID1, STORAGE_ID2, SParam1shm, SParam2shm, numOfDevices, segNumOfFreqPoints, swpCnt);

                //>>>>> This FILE pointer is part of test code used to demonstrate printing sweep results
                //>>>>> saved in shared memory into CSV file
                testfPtr = fopen("./Data/SPARAM_ReArrTest.csv", "w");

                if (avgSwpData)
                {
                    PrintSweepResToFileSegmented(testfPtr, numOfDevices, segNumOfFreqPoints, 0, resultFormat, SParam1shm, SParam2shm);
                }
                else
                {
                    for (i = 0; i < swpCnt; i++)
                    {
                        PrintSweepResToFileSegmented(testfPtr, numOfDevices, segNumOfFreqPoints, i, resultFormat, SParam1shm, SParam2shm);
                    }
                }

                fclose(testfPtr);
            }

            printf("Segmented Sweep completed\n\n");
        }
        
    }

    else
    {

        signal(SIGINT, sig_handler);


        set_conio_terminal_mode();
        pthread_create(&abortThrId, NULL, AbortSweepIntrpt, NULL); // create and run abort interrupt thread

        
        rStatus = EthernetInitFrequencySweepRawData(numOfDevices, swpTypeInput, &unitNumber[0], socketsArranged, &startF, &stopF, &stepF, ifBw,
                                                    resultFormat, swpCnt, mode, delayBtwSwp, false, saveMode);
        
        if (rStatus != MN7021aERR_NONE)
        {
            printf("Error during Frequency Sweep\n\n");
			getParamErrString(SweepParamErrString);
			printf("Param Errors: %s\n",SweepParamErrString);
        }

        signal(SIGINT, SIG_DFL);

        // re-check to make sure sweep process has really completed
        // NOTE: EthernetInitFrequencySweepRawData() function itself checks this internally before returning
        // This step is just provides optional security
        complete = CheckSweepComplete();

        if (complete == 0)
        {
            complete = 1; // reset the sweep complete flag before the next run
        }


        pthread_cancel(abortThrId); // cancel abort interrupt thread
        pthread_detach(abortThrId);

        reset_terminal_mode();

        if (rStatus == MN7021aERR_NONE)
        {
            //>>>>> This portion is test code used to demonstrate reading sweep results that was stored in shared memory
            //>>>>> subsequently printing the read sweep results into CSV file
            if (saveMode == SAVETOMEM || saveMode == SAVETOMEMANDFILE)
            {
                // ReadResultFromMemory(STORAGE_ID1, STORAGE_ID2, SParam1shm, SParam2shm, numOfDevices, startF, stopF, stepF, swpCnt);
                ReadResultFromMemoryWithExistingShm(&fd1, &fd2, SParam1shm, SParam2shm, numOfDevices, startF, stopF, stepF, swpCnt);

                //>>>>> This FILE pointer is part of test code used to demonstrate printing sweep results
                //>>>>> saved in shared memory into CSV file
                testfPtr = fopen("./Data/SPARAM_ReArrTest.csv", "w");

                if (avgSwpData)
                {
                    PrintSweepResToFile(testfPtr, numOfDevices, numOfSweepPoints, 0, resultFormat, SParam1shm, SParam2shm);
                }
                else
                {
                    for (i = 0; i < swpCnt; i++)
                    {
                        PrintSweepResToFile(testfPtr, numOfDevices, numOfSweepPoints, i, resultFormat, SParam1shm, SParam2shm);
                    }
                }

                fclose(testfPtr);
            }

            printf("Sweep completed\n\n");
        }     

    }
	//>>>>>

   
  // =============================================================
  // BATCH SWEEP v2 - position x trial loop with skip tracking
  // =============================================================
  {
    char bs_csvPath[640];
    const char *bs_subNames[4] = {"Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"};

    // ---------- Baseline sweeps ----------
    printf("\n=================================================\n");
    printf("   BASELINE - REMOVE ALL OBJECTS FROM CONTAINER\n");
    printf("=================================================\n");
    printf("Make sure the phantom/container is in its EMPTY reference\n");
    printf("state (no metal rod, no inclusions, etc.) before continuing.\n");
    printf("This will take %d baseline sweep(s) back-to-back.\n\n", bs_trialCount);
    {
      // Use bs_read_line + explicit '0' confirmation. A plain "press Enter"
      // getchar() loop would be skipped here because Keysight's last scanf
      // ("calculate active sweep time") leaves a stale newline in stdin -
      // our getchar would consume it immediately and proceed without ever
      // pausing for the user.
      if (bs_driftMode) {
        printf("  [DRIFT] Auto-proceeding to baseline (no operator confirmation).\n");
      } else {
        char bs_baseline_resp[16] = {0};
        while (1) {
          bs_read_line("Type '0' and press Enter when ready for baseline: ",
                       bs_baseline_resp, sizeof(bs_baseline_resp));
          if (bs_baseline_resp[0] == '0') break;
          printf("  Type '0' (zero) to confirm you're ready.\n");
        }
      }
    }

    for (int bs_t = 0; bs_t < bs_trialCount; bs_t++) {
      if (!bs_autoMode) {
        printf("  Baseline %d/%d - press Enter to sweep...", bs_t + 1, bs_trialCount);
        fflush(stdout);
        { int c; while ((c = getchar()) != '\n' && c != EOF) {} }
      } else {
        printf("  Baseline %d/%d ... ", bs_t + 1, bs_trialCount);
        fflush(stdout);
      }

      bs_run_one_sweep(numOfDevices, swpTypeInput, &unitNumber[0], socketsArranged,
                       &startF, &stopF, &stepF, ifBw, resultFormat, swpCnt, mode,
                       delayBtwSwp, saveMode, segSwp, segPort,
                       avgSwpData, &fd1, &fd2, &abortThrId);

      snprintf(bs_csvPath, sizeof(bs_csvPath), "%s/baseline_T%02d.csv",
               bs_sessionFolder, bs_t + 1);
      bs_write_csv(bs_csvPath, numOfDevices,
                   segSwp == 1 ? segNumOfFreqPoints : numOfSweepPoints,
                   swpCnt, resultFormat, segSwp, avgSwpData);
      printf("-> %s\n", bs_csvPath + strlen(bs_sessionFolder) + 1);

      if (bs_autoMode && bs_t < bs_trialCount - 1) {
        usleep((useconds_t)(bs_autoDelaySec * 1e6));
      }
    }

    // ---------- Position x trial loop ----------
    printf("\n=================================================\n");
    printf("   DATA COLLECTION  (%d positions x %d trials)\n",
           bs_numPositions, bs_trialCount);
    printf("=================================================\n");
    printf("\nTip: at any position prompt, type 'r' + Enter to redo the\n");
    printf("     previous position's measurement.\n");

    int bs_skipped = 0;
    int bs_pIdx = 0;
    // Track the most recently processed position so the user can press 'r'
    // at the next position's prompt to redo it. bs_prevRow == 0 means no
    // previous position yet (we're at the very first iteration).
    int bs_prevRow = 0, bs_prevCol = 0, bs_prevSub = 0;
    char bs_prevLabel[BS_LABEL_LEN] = {0};
    for (int bs_ri = 0; bs_ri < bs_numMeasureRows; bs_ri++) {
      int bs_row = bs_measureRows[bs_ri];
      for (int bs_ci = 0; bs_ci < bs_numMeasureCols; bs_ci++) {
        int bs_col = bs_measureCols[bs_ci];
        for (int bs_sp = 1; bs_sp <= 4; bs_sp++) {
          bs_pIdx++;
          double bs_x, bs_y;
          bs_grid_to_physical(bs_row, bs_col, bs_sp,
                              bs_cellSizeInch, bs_dividerInch, &bs_x, &bs_y);
          char bs_label[BS_LABEL_LEN];
          snprintf(bs_label, sizeof(bs_label), "R%dC%dP%d", bs_row, bs_col, bs_sp);

          printf("\n--------------------------------------------\n");
          printf("POSITION %d/%d   R%d C%d %s (P%d)\n",
                 bs_pIdx, bs_numPositions, bs_row, bs_col,
                 bs_subNames[bs_sp - 1], bs_sp);
          printf("  Physical: (%.2f, %.2f) inches\n", bs_x, bs_y);
          printf("--------------------------------------------\n");

          // Check auto-skip list
          int bs_autoSkipped = (bs_useAutoSkip &&
                                bs_is_in_skip_list(bs_label,
                                                   bs_autoSkipList,
                                                   bs_numAutoSkips));
          char bs_resp[16] = {0};

          // ---- DRIFT-MODE BRANCH ----
          // Fully unattended path: honor the saved skip list silently, or wait
          // the configured inter-position delay to simulate an operator move.
          // Falls through to the shared trial loop below.
          if (bs_driftMode) {
            if (bs_autoSkipped) {
              if (bs_numSessionSkips < BS_MAX_SKIPS) {
                strncpy(bs_sessionSkips[bs_numSessionSkips], bs_label, BS_LABEL_LEN - 1);
                bs_sessionSkips[bs_numSessionSkips][BS_LABEL_LEN - 1] = '\0';
                bs_numSessionSkips++;
              }
              bs_skipped++;
              printf("  [DRIFT] auto-skipped %s (in saved skip list)\n", bs_label);
              bs_prevRow = bs_row; bs_prevCol = bs_col; bs_prevSub = bs_sp;
              strncpy(bs_prevLabel, bs_label, BS_LABEL_LEN - 1);
              bs_prevLabel[BS_LABEL_LEN - 1] = '\0';
              continue;   // next position (skip trial loop)
            } else {
              printf("  [DRIFT] simulated inter-position delay %.2fs ...\n",
                     bs_driftMoveDelaySec);
              usleep((useconds_t)(bs_driftMoveDelaySec * 1e6));
            }
          } else
          if (bs_autoSkipped) {
            // Loop the auto-skip prompt so 'r' (redo previous) re-prompts
            // instead of being treated as "anything not '0'" -> confirm skip.
            while (1) {
              bs_read_line("  ** AUTO-SKIPPED (saved). Type '0' to measure anyway, or Enter to confirm skip: ",
                           bs_resp, sizeof(bs_resp));
              if (bs_resp[0] == 'r' || bs_resp[0] == 'R') {
                // -------- REDO PREVIOUS POSITION (same as regular path) --------
                if (bs_prevRow == 0) {
                  printf("  No previous position to redo (you're on the first position).\n");
                  continue;
                }

                double bs_redoX, bs_redoY;
                bs_grid_to_physical(bs_prevRow, bs_prevCol, bs_prevSub,
                                    bs_cellSizeInch, bs_dividerInch,
                                    &bs_redoX, &bs_redoY);

                printf("\n--- REDO / MODIFY PREVIOUS POSITION ---\n");
                printf("Previous position:\n");
                printf("  %s  =  R%d C%d %s (P%d)\n",
                       bs_prevLabel, bs_prevRow, bs_prevCol,
                       bs_subNames[bs_prevSub - 1], bs_prevSub);
                printf("  Physical: (%.2f, %.2f) inches\n", bs_redoX, bs_redoY);
                printf("Options:\n");
                printf("  '0' = remeasure (overwrite existing CSVs)\n");
                printf("  's' = convert to skip (delete existing CSVs, mark as skipped)\n");
                printf("  'c' = cancel (no changes)\n");
                printf("---------------------------------------\n");

                char bs_redoResp[16] = {0};
                while (1) {
                  bs_read_line("Choice: ",
                               bs_redoResp, sizeof(bs_redoResp));
                  if (bs_redoResp[0] == '0') break;
                  if (bs_redoResp[0] == 's' || bs_redoResp[0] == 'S') break;
                  if (bs_redoResp[0] == 'c' || bs_redoResp[0] == 'C') break;
                  printf("  Type '0' to remeasure, 's' to skip, or 'c' to cancel.\n");
                }

                if (bs_redoResp[0] == 'c' || bs_redoResp[0] == 'C') {
                  printf("  Cancelled. Back to %s.\n", bs_label);
                  continue;  // re-prompt the auto-skip
                }

                if (bs_redoResp[0] == 's' || bs_redoResp[0] == 'S') {
                  // Convert previous position to a skip: delete any CSVs that
                  // exist for it, then mark it as skipped (if not already).
                  int bs_deleted = 0;
                  for (int bs_dt = 0; bs_dt < bs_trialCount; bs_dt++) {
                    char bs_delPath[640];
                    snprintf(bs_delPath, sizeof(bs_delPath), "%s/%s_T%02d.csv",
                             bs_sessionFolder, bs_prevLabel, bs_dt + 1);
                    if (remove(bs_delPath) == 0) bs_deleted++;
                  }

                  int bs_already_skipped = 0;
                  for (int i = 0; i < bs_numSessionSkips; i++) {
                    if (strcmp(bs_sessionSkips[i], bs_prevLabel) == 0) {
                      bs_already_skipped = 1;
                      break;
                    }
                  }
                  if (!bs_already_skipped) {
                    if (bs_numSessionSkips < BS_MAX_SKIPS) {
                      strncpy(bs_sessionSkips[bs_numSessionSkips], bs_prevLabel, BS_LABEL_LEN - 1);
                      bs_sessionSkips[bs_numSessionSkips][BS_LABEL_LEN - 1] = '\0';
                      bs_numSessionSkips++;
                    }
                    bs_skipped++;
                  }
                  if (bs_deleted > 0) {
                    printf("  Deleted %d CSV file(s) and marked %s as skipped.\n",
                           bs_deleted, bs_prevLabel);
                  } else if (bs_already_skipped) {
                    printf("  %s was already marked as skipped (no changes).\n", bs_prevLabel);
                  } else {
                    printf("  %s marked as skipped (no CSVs were on disk).\n", bs_prevLabel);
                  }
                  printf("  Back to %s.\n", bs_label);
                  continue;  // re-prompt the auto-skip
                }

                // Otherwise '0' -> run trials below
                printf("  Redoing %s (%d trials)...\n",
                       bs_prevLabel, bs_trialCount);
                for (int bs_rt = 0; bs_rt < bs_trialCount; bs_rt++) {
                  if (!bs_autoMode) {
                    printf("  [Redo] Trial %d/%d - press Enter to sweep...",
                           bs_rt + 1, bs_trialCount);
                    fflush(stdout);
                    { int c; while ((c = getchar()) != '\n' && c != EOF) {} }
                  } else {
                    printf("  [Redo] Trial %d/%d ... ", bs_rt + 1, bs_trialCount);
                    fflush(stdout);
                  }

                  bs_run_one_sweep(numOfDevices, swpTypeInput, &unitNumber[0],
                                   socketsArranged, &startF, &stopF, &stepF, ifBw,
                                   resultFormat, swpCnt, mode, delayBtwSwp,
                                   saveMode, segSwp, segPort,
                                   avgSwpData, &fd1, &fd2, &abortThrId);

                  snprintf(bs_csvPath, sizeof(bs_csvPath),
                           "%s/%s_T%02d.csv", bs_sessionFolder,
                           bs_prevLabel, bs_rt + 1);
                  bs_write_csv(bs_csvPath, numOfDevices,
                               segSwp == 1 ? segNumOfFreqPoints : numOfSweepPoints,
                               swpCnt, resultFormat, segSwp, avgSwpData);
                  printf("-> %s_T%02d.csv\n", bs_prevLabel, bs_rt + 1);

                  if (bs_autoMode && bs_rt < bs_trialCount - 1) {
                    usleep((useconds_t)(bs_autoDelaySec * 1e6));
                  }
                }

                // If the redone position was in the session skip list, drop it
                int bs_new_count = 0, bs_was_skipped = 0;
                for (int i = 0; i < bs_numSessionSkips; i++) {
                  if (strcmp(bs_sessionSkips[i], bs_prevLabel) == 0) {
                    bs_was_skipped = 1;
                    continue;
                  }
                  if (bs_new_count != i) {
                    strncpy(bs_sessionSkips[bs_new_count],
                            bs_sessionSkips[i], BS_LABEL_LEN - 1);
                    bs_sessionSkips[bs_new_count][BS_LABEL_LEN - 1] = '\0';
                  }
                  bs_new_count++;
                }
                bs_numSessionSkips = bs_new_count;
                if (bs_was_skipped && bs_skipped > 0) {
                  bs_skipped--;
                  printf("  (%s removed from skip list since it now has real data.)\n",
                         bs_prevLabel);
                }

                printf("  Redo of %s complete. Back to %s.\n",
                       bs_prevLabel, bs_label);
                continue;  // re-prompt the auto-skip
                // ---------------------------------------------------------------
              }
              break;  // anything else (0, Enter, etc.): exit the prompt loop
            }
            if (bs_resp[0] != '0') {
              // Add to session-skipped list
              if (bs_numSessionSkips < BS_MAX_SKIPS) {
                strncpy(bs_sessionSkips[bs_numSessionSkips], bs_label, BS_LABEL_LEN - 1);
                bs_sessionSkips[bs_numSessionSkips][BS_LABEL_LEN - 1] = '\0';
                bs_numSessionSkips++;
              }
              bs_skipped++;
              printf("  SKIPPED %s\n", bs_label);
              // Update prev-position tracker so 'r' at the next position can
              // redo this one (yes, you can redo a skipped position).
              bs_prevRow = bs_row; bs_prevCol = bs_col; bs_prevSub = bs_sp;
              strncpy(bs_prevLabel, bs_label, BS_LABEL_LEN - 1);
              bs_prevLabel[BS_LABEL_LEN - 1] = '\0';
              continue;
            } else {
              printf("  Override: will measure this position.\n");
            }
          } else {
            while (1) {
              bs_read_line("Place object. Type '0' to confirm, 's' to skip: ",
                           bs_resp, sizeof(bs_resp));
              if (bs_resp[0] == '0') break;
              if (bs_resp[0] == 's' || bs_resp[0] == 'S') break;
              if (bs_resp[0] == 'r' || bs_resp[0] == 'R') {
                // -------- REDO PREVIOUS POSITION --------
                if (bs_prevRow == 0) {
                  printf("  No previous position to redo (you're on the first position).\n");
                  continue;  // re-prompt for current position
                }

                double bs_redoX, bs_redoY;
                bs_grid_to_physical(bs_prevRow, bs_prevCol, bs_prevSub,
                                    bs_cellSizeInch, bs_dividerInch,
                                    &bs_redoX, &bs_redoY);

                printf("\n--- REDO / MODIFY PREVIOUS POSITION ---\n");
                printf("Previous position:\n");
                printf("  %s  =  R%d C%d %s (P%d)\n",
                       bs_prevLabel, bs_prevRow, bs_prevCol,
                       bs_subNames[bs_prevSub - 1], bs_prevSub);
                printf("  Physical: (%.2f, %.2f) inches\n", bs_redoX, bs_redoY);
                printf("Options:\n");
                printf("  '0' = remeasure (overwrite existing CSVs)\n");
                printf("  's' = convert to skip (delete existing CSVs, mark as skipped)\n");
                printf("  'c' = cancel (no changes)\n");
                printf("---------------------------------------\n");

                char bs_redoResp[16] = {0};
                while (1) {
                  bs_read_line("Choice: ",
                               bs_redoResp, sizeof(bs_redoResp));
                  if (bs_redoResp[0] == '0') break;
                  if (bs_redoResp[0] == 's' || bs_redoResp[0] == 'S') break;
                  if (bs_redoResp[0] == 'c' || bs_redoResp[0] == 'C') break;
                  printf("  Type '0' to remeasure, 's' to skip, or 'c' to cancel.\n");
                }

                if (bs_redoResp[0] == 'c' || bs_redoResp[0] == 'C') {
                  printf("  Cancelled. Back to %s.\n", bs_label);
                  continue;  // re-prompt for the original current position
                }

                if (bs_redoResp[0] == 's' || bs_redoResp[0] == 'S') {
                  // Convert previous position to a skip: delete any CSVs that
                  // exist for it, then mark it as skipped (if not already).
                  int bs_deleted = 0;
                  for (int bs_dt = 0; bs_dt < bs_trialCount; bs_dt++) {
                    char bs_delPath[640];
                    snprintf(bs_delPath, sizeof(bs_delPath), "%s/%s_T%02d.csv",
                             bs_sessionFolder, bs_prevLabel, bs_dt + 1);
                    if (remove(bs_delPath) == 0) bs_deleted++;
                  }

                  int bs_already_skipped = 0;
                  for (int i = 0; i < bs_numSessionSkips; i++) {
                    if (strcmp(bs_sessionSkips[i], bs_prevLabel) == 0) {
                      bs_already_skipped = 1;
                      break;
                    }
                  }
                  if (!bs_already_skipped) {
                    if (bs_numSessionSkips < BS_MAX_SKIPS) {
                      strncpy(bs_sessionSkips[bs_numSessionSkips], bs_prevLabel, BS_LABEL_LEN - 1);
                      bs_sessionSkips[bs_numSessionSkips][BS_LABEL_LEN - 1] = '\0';
                      bs_numSessionSkips++;
                    }
                    bs_skipped++;
                  }
                  if (bs_deleted > 0) {
                    printf("  Deleted %d CSV file(s) and marked %s as skipped.\n",
                           bs_deleted, bs_prevLabel);
                  } else if (bs_already_skipped) {
                    printf("  %s was already marked as skipped (no changes).\n", bs_prevLabel);
                  } else {
                    printf("  %s marked as skipped (no CSVs were on disk).\n", bs_prevLabel);
                  }
                  printf("  Back to %s.\n", bs_label);
                  continue;  // re-prompt for the original current position
                }

                // Otherwise '0' -> run trials below
                printf("  Redoing %s (%d trials)...\n",
                       bs_prevLabel, bs_trialCount);
                for (int bs_rt = 0; bs_rt < bs_trialCount; bs_rt++) {
                  if (!bs_autoMode) {
                    printf("  [Redo] Trial %d/%d - press Enter to sweep...",
                           bs_rt + 1, bs_trialCount);
                    fflush(stdout);
                    { int c; while ((c = getchar()) != '\n' && c != EOF) {} }
                  } else {
                    printf("  [Redo] Trial %d/%d ... ", bs_rt + 1, bs_trialCount);
                    fflush(stdout);
                  }

                  bs_run_one_sweep(numOfDevices, swpTypeInput, &unitNumber[0],
                                   socketsArranged, &startF, &stopF, &stepF, ifBw,
                                   resultFormat, swpCnt, mode, delayBtwSwp,
                                   saveMode, segSwp, segPort,
                                   avgSwpData, &fd1, &fd2, &abortThrId);

                  snprintf(bs_csvPath, sizeof(bs_csvPath),
                           "%s/%s_T%02d.csv", bs_sessionFolder,
                           bs_prevLabel, bs_rt + 1);
                  bs_write_csv(bs_csvPath, numOfDevices,
                               segSwp == 1 ? segNumOfFreqPoints : numOfSweepPoints,
                               swpCnt, resultFormat, segSwp, avgSwpData);
                  printf("-> %s_T%02d.csv\n", bs_prevLabel, bs_rt + 1);

                  if (bs_autoMode && bs_rt < bs_trialCount - 1) {
                    usleep((useconds_t)(bs_autoDelaySec * 1e6));
                  }
                }

                // If the redone position was in the session skip list, drop
                // it (since we just measured it for real) and decrement the
                // skipped count.
                int bs_new_count = 0, bs_was_skipped = 0;
                for (int i = 0; i < bs_numSessionSkips; i++) {
                  if (strcmp(bs_sessionSkips[i], bs_prevLabel) == 0) {
                    bs_was_skipped = 1;
                    continue;
                  }
                  if (bs_new_count != i) {
                    strncpy(bs_sessionSkips[bs_new_count],
                            bs_sessionSkips[i], BS_LABEL_LEN - 1);
                    bs_sessionSkips[bs_new_count][BS_LABEL_LEN - 1] = '\0';
                  }
                  bs_new_count++;
                }
                bs_numSessionSkips = bs_new_count;
                if (bs_was_skipped && bs_skipped > 0) {
                  bs_skipped--;
                  printf("  (%s removed from skip list since it now has real data.)\n",
                         bs_prevLabel);
                }

                printf("  Redo of %s complete. Back to %s.\n",
                       bs_prevLabel, bs_label);
                continue;  // re-prompt for the original current position
                // ---------------------------------------
              }
              printf("  Type '0' to confirm or 's' to skip.\n");
            }
            if (bs_resp[0] == 's' || bs_resp[0] == 'S') {
              if (bs_numSessionSkips < BS_MAX_SKIPS) {
                strncpy(bs_sessionSkips[bs_numSessionSkips], bs_label, BS_LABEL_LEN - 1);
                bs_sessionSkips[bs_numSessionSkips][BS_LABEL_LEN - 1] = '\0';
                bs_numSessionSkips++;
              }
              bs_skipped++;
              printf("  SKIPPED %s.\n", bs_label);
              // Update prev-position tracker so 'r' at the next position can
              // redo this one (you can redo a manually-skipped position too).
              bs_prevRow = bs_row; bs_prevCol = bs_col; bs_prevSub = bs_sp;
              strncpy(bs_prevLabel, bs_label, BS_LABEL_LEN - 1);
              bs_prevLabel[BS_LABEL_LEN - 1] = '\0';
              continue;
            }
          }

          // Measure trialCount times at this position
          for (int bs_t = 0; bs_t < bs_trialCount; bs_t++) {
            if (!bs_autoMode) {
              char bs_pr[64];
              snprintf(bs_pr, sizeof(bs_pr),
                       "  Trial %d/%d - press Enter to sweep...",
                       bs_t + 1, bs_trialCount);
              printf("%s", bs_pr);
              fflush(stdout);
              { int c; while ((c = getchar()) != '\n' && c != EOF) {} }
            } else {
              printf("  Trial %d/%d ... ", bs_t + 1, bs_trialCount);
              fflush(stdout);
            }

            bs_run_one_sweep(numOfDevices, swpTypeInput, &unitNumber[0],
                             socketsArranged, &startF, &stopF, &stepF, ifBw,
                             resultFormat, swpCnt, mode, delayBtwSwp,
                             saveMode, segSwp, segPort,
                             avgSwpData, &fd1, &fd2, &abortThrId);

            snprintf(bs_csvPath, sizeof(bs_csvPath),
                     "%s/%s_T%02d.csv", bs_sessionFolder,
                     bs_label, bs_t + 1);
            bs_write_csv(bs_csvPath, numOfDevices,
                         segSwp == 1 ? segNumOfFreqPoints : numOfSweepPoints,
                         swpCnt, resultFormat, segSwp, avgSwpData);
            printf("-> %s_T%02d.csv\n", bs_label, bs_t + 1);

            if (bs_autoMode && bs_t < bs_trialCount - 1) {
              usleep((useconds_t)(bs_autoDelaySec * 1e6));
            }
          }

          // Update prev-position tracker so 'r' at the next position can
          // redo this one.
          bs_prevRow = bs_row; bs_prevCol = bs_col; bs_prevSub = bs_sp;
          strncpy(bs_prevLabel, bs_label, BS_LABEL_LEN - 1);
          bs_prevLabel[BS_LABEL_LEN - 1] = '\0';
        }
      }
    }

    printf("\n=================================================\n");
    printf("   DATA COLLECTION COMPLETE\n");
    printf("=================================================\n");
    printf("Session folder:    %s\n", bs_sessionFolder);
    printf("Positions taken:   %d / %d\n", bs_numPositions - bs_skipped, bs_numPositions);
    printf("Positions skipped: %d this session\n", bs_skipped);

    // ---------- Offer to save session skips ----------
    if (bs_numSessionSkips > 0 && !bs_driftMode) {
      printf("\nSkipped positions this session:\n");
      for (int ii = 0; ii < bs_numSessionSkips; ii++)
        printf("  %s\n", bs_sessionSkips[ii]);
      char saveResp[16];
      bs_read_line("Save these skips (merged with existing) to model's skip list? [Y/n]: ",
                   saveResp, sizeof(saveResp));
      if (!(saveResp[0] == 'n' || saveResp[0] == 'N')) {
        bs_merge_and_save_skips(bs_modelName,
                                bs_autoSkipList, bs_numAutoSkips,
                                bs_sessionSkips, bs_numSessionSkips);
        printf("  Skip list updated for '%s'.\n", bs_modelName);
      }
    } else if (bs_driftMode && bs_numSessionSkips > 0) {
      printf("\n[DRIFT MODE] Skipped positions this session: %d\n"
             "  (not saving to model's persistent skip list)\n",
             bs_numSessionSkips);
    }

    // ---------- Write a session README.md tailored to what was captured ----------
    {
      char bs_readmePath[640];
      snprintf(bs_readmePath, sizeof(bs_readmePath), "%s/README.md", bs_sessionFolder);
      const char *bs_folderBasename = strrchr(bs_sessionFolder, '/');
      bs_folderBasename = bs_folderBasename ? bs_folderBasename + 1 : bs_sessionFolder;
      bs_write_readme(bs_readmePath, bs_modelName, bs_antennaName, bs_objectName,
                      bs_operatorName, bs_gridRows, bs_gridCols,
                      bs_measureRows, bs_numMeasureRows,
                      bs_measureCols, bs_numMeasureCols,
                      bs_cellSizeInch, bs_dividerInch,
                      bs_trialCount, bs_numPositions, bs_autoMode,
                      bs_numPositions - bs_skipped, bs_skipped,
                      bs_sessionSkips, bs_numSessionSkips,
                      bs_folderBasename,
                      bs_notes);
      printf("Wrote session README: %s/README.md\n", bs_sessionFolder);
    }

    printf("=================================================\n\n");

    // Cleanup
    if (saveMode == SAVETOMEM || saveMode == SAVETOMEMANDFILE) {
      fd1 = shm_unlink(STORAGE_ID1);
      if (fd1 == -1) perror("unlink STORAGE_ID1");
      fd2 = shm_unlink(STORAGE_ID2);
      if (fd2 == -1) perror("unlink STORAGE_ID2");
    }
    rStatus = DisconnectEthernet(numOfDevices, socketsArranged);
    return 0;
  }
  // =============================================================
  // end BATCH SWEEP v2 position-trial loop
  // =============================================================


  return 0;
}



/*************************************************************************************************************************************
   Function Name :	GetIFBWEnumStringFromEnum
   Description :	Get the enum string based on the input IFBandwidth enum
   Arguments : 		IFBandwidth bw
   Returns :		Corresponding string representation of the input enum
   Note :          	NONE
**************************************************************************************************************************************/
const char* GetIFBWEnumStringFromEnum(unsigned short bw)
{
   switch (bw)
   {
      case 1: return "6MHz";
      case 2: return "3MHz";
      case 4: return "1.5MHz";
      case 8: return "750kHz";
      case 16: return "390kHz";
      case 32: return "195kHz";
      case 64: return "100kHz";
      case 128: return "50kHz";
      case 256: return "25kHz";
      case 512: return "12kHz";
      case 1024: return "6kHz";
      case 2048: return "3kHz";
      case 4096: return "1.5kHz";
   }

   return "NULL";
}

/*************************************************************************************************************************************
   Function Name :	GetIFBWEnumStringFromUInt16
   Description :	Get the enum string based on the input unsigned short bandwidth parameter
   Arguments : 		unsigned short bw
   Returns :		Corresponding string representation of the input unsigned short bw parameter
   Note :          	NONE
**************************************************************************************************************************************/
const char* GetIFBWEnumStringFromUInt16(unsigned short bw)
{
   switch (bw)
   {
      case 1: return "6MHz";
      case 2: return "3MHz";
      case 4: return "1.5MHz";
      case 8: return "750kHz";
      case 16: return "390kHz";
      case 32: return "195kHz";
      case 64: return "100kHz";
      case 128: return "50kHz";
      case 256: return "25kHz";
      case 512: return "12kHz";
      case 1024: return "6kHz";
      case 2048: return "3kHz";
      case 4096: return "1.5kHz";
   }

   return "NULL";
}

/*************************************************************************************************************************************
   Function Name :	CompareAllUnitFrimwareVer
   Description :	Compares firmware versions of all units against each other to check if all units are having the same version
   Arguments : 		int devCount, char allFwVer[4][8]
   Returns :		true if all units firmware version matches; false otherwise
   Note :          	NONE
**************************************************************************************************************************************/
bool CompareAllUnitFrimwareVer(int devCount, char allFwVer[4][8])
{
    bool diff = false;
    char tempFwVer[8];
    int tmp = 0;
    int i = 0;
    int j = 0;
    int comp = 0;

    for (i = 0; i < devCount; i++)
    {
        for (tmp = 0; tmp < 8; tmp++)
        {
            tempFwVer[tmp] = allFwVer[i][tmp];
        }

        for (j = 0; j < devCount; j++)
		{
			if (i == j)
            {
                continue;
            }
            else
            {
                for (comp = 0; comp < 8; comp++)
                {
                    if (tempFwVer[comp] != allFwVer[j][comp])
                    {
                        diff = true;
                    }
                }
            }
		}
    }

    return diff;
}

/*************************************************************************************************************************************
   Function Name :	CheckSweepPoints
   Description :	Checks that the start frequency, stop frequency and frequency step does not result in an over range of a maximum
                    number of sweep points of 500 (1000 for single or two units).
   Arguments : 		double* start, double* stop, double* step
   Returns :		NONE
   Note :          	This is an interactive function where it will prompt the user to input another set of values if the initial i
                    input yields more than 500 sweep points
**************************************************************************************************************************************/
void CheckSweepPoints(double* start, double* stop, double* step)
{
    int verdict = -1;
    double tempStart = *start;
    double tempStop = *stop;
    double tempStep = *step;
    int swpPts = ((tempStop - tempStart) / tempStep) + 1;
    int choice = 1;
    int maxSwpPts = 500;

    if (fileNumOfHunter <= 2)
        maxSwpPts = 1000;

    while (swpPts > maxSwpPts || swpPts < 2)
    {
        printf("********* WARNING *********\n");
        printf("The sweep points is %d\n",swpPts);
        printf("The input start frequency, stop frequency and frequency step yields more than the maximum allowable sweep points of 500 (1000 for single or 2 units).\n");
        printf("Please choose [1] to use the default sweep frequencies -->\n");
        printf("Start frequency = 500 MHz\n");
        printf("Stop frequency = 2 GHz\n");
        printf("Frequency step = 4 MHz\n");
        printf("or choose [2] to re-input the sweep frequencies: \n");

        while ((verdict = GetChoice(&choice)) != 0)
        {
            printf("The choice was invalid...\n");
            printf("Please choose [1] to use the default sweep frequencies\n");
            printf("or choose [2] to re-input the sweep frequencies: \n");
        }

        if (choice == 2)
        {
            printf("Please input Start Frequency: ");
            GetDoubleInput(&tempStart);

            printf("Please input Stop Frequency: ");
            GetDoubleInput(&tempStop);

            printf("Please input Step Frequency: ");
            GetDoubleInput(&tempStep);

        }
        else
        {
            tempStart = 500000000;
            tempStop = 2000000000;
            tempStep = 4000000;
        }

        swpPts = ((tempStop - tempStart) / tempStep) + 1;
    }

    *start = tempStart;
    *stop = tempStop;
    *step = tempStep;
}

void lossCommunicationSignalHandler(int signum)
{
    for (int i = 0; i < fileNumOfHunter; i++)
    {
        if (!UnitsStatus[i].connectionStat)
            printf("\n%s of MN7021A is down...\n", UnitsStatus[i].ipAddress);
    }

    printf("Kindly check your MN7021A network connection or reboot the MN7021A unit...\n"); 
    exit(1);
}

void sig_handler(int signum)
{
    if (signum == SIGALRM)
    {
        printf("\nAlarm aborting...\n");
    }
    printf("\nSweep aborting...\n");

    // usleep(200000);
    EthernetClearSweepOperatives();    
    // pthread_cancel(supThrId);
}

void sig_handler2(int signum)
{
//    if (signum == SIGALRM)
//    {
        printf("\nTimeout...\n");
//    }
//    printf("\nSweep aborting...\n");

    timeoutTrig = true;

    int succ = 0;
    succ = succ;
    succ = pthread_kill(parent, SIGINT);

//    EthernetClearSweepOperatives();
}




void *AbortSweepIntrpt(void* arg)
{
    char c = '0';
    int succ = 0;
    succ = succ;
    int s = 0;
    s = s;

    printf("Hit 'X' to abort sweep...\n");

    s = pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
	s = pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);



    do
    {
//        ualarm(100000, 100000);
//        signal(SIGALRM, got_alarm);

//        c = getchar();

        while (!kboardhit())
        {

        }
        c = getkey();

        if(!isAbortAllowed)
        {
            printf("Abort is not allowed during data crunching. \n");
        }

        if (c == 'X' && !GetSweepEndFlag() && isAbortAllowed)
        {
            reset_terminal_mode();

            alarm(0);
            signal(SIGALRM, SIG_DFL);
            
            succ = pthread_kill(parent, SIGINT);
            break;
        }
    }while (!GetSweepEndFlag());

    // pthread_cancel(supThrId);
    // alarm(0);
    // signal(SIGALRM, SIG_DFL);

    return 0;
}


void got_alarm(int sig)
{
//    printf("got signal %d\n", sig);
}

void ComputeRecSupvParam(int sweepCount, int* intvlmicroS)
{
    int temp;

    if (sweepCount >= 10)
    {
        temp = (int)(floor(sweepCount * 0.1)) * 100000;
    }
    else
    {
        temp = 100000;
    }

    *intvlmicroS = temp;
}

/*************************************************************************************************************************************
   Function Name :	GetInputDataCorr
   Description :	Validate user input for Y/y or N/n
   Arguments : 		int rStatus
   Returns :		Y/y to exit the program; N/n to continue without cal data correction
   Note :          	NONE
**************************************************************************************************************************************/
void GetInputDataCorr(int rStatus)
{
    int verdict;
    char input;
    int numOfDevices = fileNumOfHunter;
    int deviceSocket[4]; 

    if (rStatus != 0)
    {
        printf("Do you want to exit the application and run calibration data correction first [Y/N]? ");
        while ((verdict = GetInput(&input)) != 0)
        {
            printf("Please input 'Y' or 'y' to exit the application\n");
            printf("or input 'N' or 'n' to continue without cal data correction: ");
        }
        if ((input == 'Y') || (input == 'y'))
        {
            DisconnectEthernet(numOfDevices, deviceSocket);
            exit(0);
        }
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// =============================================================
// BATCH SWEEP v2 helpers
// =============================================================
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>

void bs_read_line(const char *prompt, char *out, size_t outSize) {
  printf("%s", prompt);
  fflush(stdout);
  if (fgets(out, (int)outSize, stdin) == NULL) {
    out[0] = '\0';
    return;
  }
  size_t L = strlen(out);
  while (L > 0 && (out[L-1] == '\n' || out[L-1] == '\r')) out[--L] = '\0';
}

int bs_read_int(const char *prompt, int defaultVal) {
  char buf[64];
  bs_read_line(prompt, buf, sizeof(buf));
  if (buf[0] == '\0') return defaultVal;
  return atoi(buf);
}

double bs_read_double(const char *prompt, double defaultVal) {
  char buf[64];
  bs_read_line(prompt, buf, sizeof(buf));
  if (buf[0] == '\0') return defaultVal;
  return atof(buf);
}

int bs_read_int_list(const char *prompt, int *out, int maxCount) {
  char buf[512];
  bs_read_line(prompt, buf, sizeof(buf));
  int n = 0;
  char *tok = strtok(buf, " ,\t");
  while (tok && n < maxCount) {
    out[n++] = atoi(tok);
    tok = strtok(NULL, " ,\t");
  }
  return n;
}

void bs_sanitize(const char *in, char *out, size_t outSize) {
  size_t i = 0, j = 0;
  while (in[i] && j + 1 < outSize) {
    char c = in[i];
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '_' || c == '-') {
      out[j++] = c;
    } else if (c == ' ') {
      out[j++] = '_';
    }
    i++;
  }
  out[j] = '\0';
  if (j == 0) { strncpy(out, "Unnamed", outSize - 1); out[outSize - 1] = '\0'; }
}

void bs_ensure_dir(const char *path) {
  DIR *d = opendir(path);
  if (d) { closedir(d); return; }
  if (errno == ENOENT) mkdir(path, 0777);
}

void bs_grid_to_physical(int row, int col, int subPos,
                         double cellSize, double dividerThick,
                         double *xInch, double *yInch) {
  double halfDiv = dividerThick / 2.0;
  double offset  = (cellSize / 2.0) - halfDiv;
  double dx = 0.0, dy = 0.0;
  switch (subPos) {
    case 1: dx = -offset; dy = -offset; break;
    case 2: dx = +offset; dy = -offset; break;
    case 3: dx = +offset; dy = +offset; break;
    case 4: dx = -offset; dy = +offset; break;
    default: break;
  }
  *xInch = (col - 0.5) * cellSize + dx;
  *yInch = (row - 0.5) * cellSize + dy;
}

void bs_write_metadata(const char *path, const char *modelName,
                       const char *antennaName, const char *objectName,
                       const char *operatorName, int gridRows, int gridCols,
                       int *measureRows, int numMeasureRows,
                       int *measureCols, int numMeasureCols,
                       double cellSizeInch, double dividerInch,
                       int trialCount, int numPositions, int autoMode,
                       const char *notes) {
  FILE *f = fopen(path, "w");
  if (!f) return;
  time_t t = time(NULL);
  struct tm tm = *localtime(&t);
  fprintf(f, "Hunter VNA Batch Sweep - Session Metadata\n");
  fprintf(f, "==========================================\n\n");
  fprintf(f, "Timestamp:  %04d-%02d-%02d %02d:%02d:%02d\n",
          tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
          tm.tm_hour, tm.tm_min, tm.tm_sec);
  fprintf(f, "Operator:   %s\n", operatorName);
  fprintf(f, "Model:      %s\n", modelName);
  fprintf(f, "Antenna:    %s\n", antennaName);
  fprintf(f, "Object:     %s\n", objectName);
  if (notes && notes[0] != '\0') {
    fprintf(f, "Notes:      %s\n", notes);
  }
  fprintf(f, "\nGrid Configuration\n");
  fprintf(f, "  Total grid:    %d rows x %d cols\n", gridRows, gridCols);
  fprintf(f, "  Measured rows:");
  for (int i = 0; i < numMeasureRows; i++) fprintf(f, " %d", measureRows[i]);
  fprintf(f, "\n  Measured cols:");
  for (int i = 0; i < numMeasureCols; i++) fprintf(f, " %d", measureCols[i]);
  fprintf(f, "\n  Cell size:     %.3f inches\n", cellSizeInch);
  fprintf(f, "  Divider thick: %.3f inches\n", dividerInch);
  fprintf(f, "\nMeasurement Plan\n");
  fprintf(f, "  Positions:           %d\n", numPositions);
  fprintf(f, "  Trials per position: %d\n", trialCount);
  fprintf(f, "  Total sweeps:        %d (+ %d baseline)\n",
          numPositions * trialCount, trialCount);
  fprintf(f, "  Mode:                %s\n", autoMode ? "AUTOMATIC" : "INTERACTIVE");
  fprintf(f, "  Antennas in use:     %d (ports 1..%d; only those S-parameters saved)\n",
          g_bs_numAntennas, g_bs_numAntennas);
  fprintf(f, "\nFile Naming\n");
  fprintf(f, "  baseline_TNN.csv       baseline trial NN\n");
  fprintf(f, "  RnCmPp_TNN.csv         row n, col m, sub-position p (1-4), trial NN\n");
  fprintf(f, "    P1=Top-Left  P2=Top-Right  P3=Bottom-Right  P4=Bottom-Left\n");
  fclose(f);
}

// Write a human-readable README.md describing the session that was just
// captured. Tailored with the actual model/antenna/object, what positions
// were measured, what was skipped, the CSV format, and how to load the data.
// Designed so that if someone receives just this session folder (without
// the rest of the Hunter install) they have enough info to use the data.
void bs_write_readme(const char *path, const char *modelName,
                     const char *antennaName, const char *objectName,
                     const char *operatorName, int gridRows, int gridCols,
                     int *measureRows, int numMeasureRows,
                     int *measureCols, int numMeasureCols,
                     double cellSizeInch, double dividerInch,
                     int trialCount, int numPositions, int autoMode,
                     int positionsTaken, int positionsSkipped,
                     char sessionSkips[][BS_LABEL_LEN], int numSessionSkips,
                     const char *sessionFolderName,
                     const char *notes) {
  FILE *f = fopen(path, "w");
  if (!f) return;
  time_t t = time(NULL);
  struct tm tm = *localtime(&t);

  fprintf(f, "# Hunter VNA Batch Sweep Session\n\n");
  fprintf(f, "Session folder: `%s`  \n", sessionFolderName);
  fprintf(f, "Completed: %04d-%02d-%02d %02d:%02d:%02d\n\n",
          tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
          tm.tm_hour, tm.tm_min, tm.tm_sec);

  fprintf(f, "## Session details\n\n");
  fprintf(f, "- **Operator:** %s\n", operatorName);
  fprintf(f, "- **Model:** %s\n", modelName);
  fprintf(f, "- **Antenna:** %s\n", antennaName);
  fprintf(f, "- **Object measured:** %s\n", objectName);
  fprintf(f, "- **Mode:** %s\n", autoMode ? "Automatic" : "Interactive");
  fprintf(f, "- **Antennas in use:** %d (ports 1..%d only; other ports' S-parameters were stripped from the saved CSVs)\n\n",
          g_bs_numAntennas, g_bs_numAntennas);

  // Notes section: only emit if the user typed something.
  if (notes && notes[0] != '\0') {
    fprintf(f, "## Notes\n\n");
    fprintf(f, "%s\n\n", notes);
  }

  fprintf(f, "## Grid configuration\n\n");
  fprintf(f, "- Total grid: %d rows x %d columns\n", gridRows, gridCols);
  fprintf(f, "- Measured rows:");
  for (int i = 0; i < numMeasureRows; i++) fprintf(f, " %d", measureRows[i]);
  fprintf(f, "\n- Measured cols:");
  for (int i = 0; i < numMeasureCols; i++) fprintf(f, " %d", measureCols[i]);
  fprintf(f, "\n- Cell size: %.3f inches\n", cellSizeInch);
  fprintf(f, "- Divider thickness: %.3f inches\n\n", dividerInch);

  fprintf(f, "## What was actually captured\n\n");
  fprintf(f, "- Baseline trials: %d files (`baseline_T01.csv` through `baseline_T%02d.csv`)\n",
          trialCount, trialCount);
  fprintf(f, "- Positions planned: %d\n", numPositions);
  fprintf(f, "- Positions measured: %d (%d trial%s each)\n",
          positionsTaken, trialCount, trialCount == 1 ? "" : "s");
  fprintf(f, "- Positions skipped: %d\n", positionsSkipped);
  if (numSessionSkips > 0) {
    fprintf(f, "\n  Skipped positions:\n");
    for (int i = 0; i < numSessionSkips; i++) {
      fprintf(f, "  - %s\n", sessionSkips[i]);
    }
  }
  fprintf(f, "- Total sweep CSVs in this folder: %d\n\n",
          trialCount + (positionsTaken * trialCount));

  fprintf(f, "## File naming\n\n");
  fprintf(f, "- `baseline_TNN.csv` -- baseline trial NN (container empty / reference state)\n");
  fprintf(f, "- `RnCmPp_TNN.csv` -- object at row n, column m, sub-position p, trial NN\n");
  fprintf(f, "- `session_metadata.txt` -- plain-text session info written at session start\n");
  fprintf(f, "- `README.md` -- this file\n\n");

  fprintf(f, "### Sub-position legend\n\n");
  fprintf(f, "Within each grid cell, four corner positions:\n\n");
  fprintf(f, "```\n");
  fprintf(f, "+----+----+\n");
  fprintf(f, "| P1 | P2 |\n");
  fprintf(f, "+----+----+\n");
  fprintf(f, "| P4 | P3 |\n");
  fprintf(f, "+----+----+\n");
  fprintf(f, "```\n\n");
  fprintf(f, "- **P1** = Top-Left\n");
  fprintf(f, "- **P2** = Top-Right\n");
  fprintf(f, "- **P3** = Bottom-Right\n");
  fprintf(f, "- **P4** = Bottom-Left\n\n");

  fprintf(f, "## CSV format\n\n");
  fprintf(f, "Each CSV is a single frequency sweep, with one row per frequency point:\n\n");
  fprintf(f, "- **Column 0:** Frequency in Hz\n");
  fprintf(f, "- **Remaining columns:** Pairs of (Magnitude, Phase) for each S-parameter\n\n");
  {
    int N = g_bs_numAntennas;
    fprintf(f, "This session was recorded with **%d antenna%s in use (ports 1..%d)**, so only the S-parameters involving those ports are saved. The column order is:\n\n",
            N, N == 1 ? "" : "s", N);
    fprintf(f, "    Frequency, ");
    int first = 1;
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= N; i++) {
        if (!first) fprintf(f, ", ");
        fprintf(f, "S%d%d_mag, S%d%d_phase", i, j, i, j);
        first = 0;
      }
    }
    fprintf(f, "\n\n");
  }
  fprintf(f, "- Magnitude is **linear** (not dB)\n");
  fprintf(f, "- Phase is in **degrees**\n");
  fprintf(f, "- To convert to dB:  `20 * log10(magnitude)`\n");
  fprintf(f, "- To convert to complex S:  `magnitude * exp(1j * phase_deg * pi / 180)`\n\n");

  fprintf(f, "## How to view this data\n\n");
  fprintf(f, "### With plot_only.py (recommended)\n\n");
  fprintf(f, "From the Hunter application folder:\n\n");
  fprintf(f, "    ./plot_only.py path/to/%s/\n\n", sessionFolderName);
  fprintf(f, "Or run `./plot_only.py` with no args, cancel the file dialog, and pick\n");
  fprintf(f, "this folder. You will get a menu of positions; selecting one loads all\n");
  fprintf(f, "trials at that position, coherently averages them in the complex domain,\n");
  fprintf(f, "and plots reflection (S11..S44) and transmission (off-diagonal Sij).\n\n");
  fprintf(f, "### With Python directly\n\n");
  fprintf(f, "    import pandas as pd, numpy as np\n");
  fprintf(f, "    df = pd.read_csv('baseline_T01.csv')\n");
  fprintf(f, "    freq_Hz = df.iloc[:, 0].values\n");
  fprintf(f, "    s11_mag = df.iloc[:, 1].values\n");
  fprintf(f, "    s11_phase_deg = df.iloc[:, 2].values\n");
  fprintf(f, "    s11_complex = s11_mag * np.exp(1j * np.deg2rad(s11_phase_deg))\n");
  fprintf(f, "    s11_dB = 20 * np.log10(s11_mag)\n\n");

  fprintf(f, "## Notes\n\n");
  fprintf(f, "- Captured with the Keysight Hunter MN7021A 4-port VNA\n");
  fprintf(f, "- The VNA was calibrated (OSL + Through) before this session\n");
  fprintf(f, "- All measurements use the calibration in effect at session start\n");
  fprintf(f, "- If a cable was bumped during the session, later measurements may have drifted\n");
  fprintf(f, "- For per-trial outlier detection, examine individual `_TNN.csv` files before averaging\n");

  fclose(f);
}

void bs_run_one_sweep(int numOfDevices, char swpTypeInput, char *unitNumber,
                      int *socketsArranged, double *startF, double *stopF,
                      double *stepF, IFBandwidth ifBw, int resultFormat,
                      int swpCnt, SweepMode mode, int delayBtwSwp,
                      int saveMode, int segSwp, int *segPort_arg,
                      bool avgSwpData, int *fd1, int *fd2,
                      pthread_t *abortThrId) {
  MN7021aErrType rStatus = MN7021aERR_NONE;
  int complete = 1;
  (void)avgSwpData;
  (void)segPort_arg;

  // Force memory storage so we can read the data and write our own labeled
  // CSV. If config.txt's saveMode is SAVETOFILE (2), the shared memory buffers
  // never get populated and bs_write_csv writes all-zero CSVs. Force SAVETOMEM
  // here so the memory read below always works regardless of config.txt.
  saveMode = SAVETOMEM;

  signal(SIGINT, sig_handler);
  set_conio_terminal_mode();
  pthread_create(abortThrId, NULL, AbortSweepIntrpt, NULL);

  if (segSwp == 1) {
    rStatus = EthernetInitSegmentedFrequencySweepRawData(
        numOfDevices, swpTypeInput, unitNumber, socketsArranged,
        resultFormat, swpCnt, false, saveMode, segPort);
  } else {
    rStatus = EthernetInitFrequencySweepRawData(
        numOfDevices, swpTypeInput, unitNumber, socketsArranged,
        startF, stopF, stepF, ifBw, resultFormat, swpCnt, mode,
        delayBtwSwp, false, saveMode);
  }

  signal(SIGINT, SIG_DFL);
  complete = CheckSweepComplete();
  if (complete == 0) complete = 1;
  pthread_cancel(*abortThrId);
  pthread_detach(*abortThrId);
  reset_terminal_mode();

  if (rStatus != MN7021aERR_NONE) {
    printf("[WARN] Sweep returned error %d\n", (int)rStatus);
    return;
  }

  if (saveMode == SAVETOMEM || saveMode == SAVETOMEMANDFILE) {
    if (segSwp == 1) {
      ReadResultFromMemorySegmented(STORAGE_ID1, STORAGE_ID2,
                                    SParam1shm, SParam2shm,
                                    numOfDevices, segNumOfFreqPoints, swpCnt);
    } else {
      ReadResultFromMemoryWithExistingShm(fd1, fd2,
                                          SParam1shm, SParam2shm,
                                          numOfDevices, *startF, *stopF, *stepF,
                                          swpCnt);
    }
  }
}

void bs_write_csv(const char *csvPath, int numOfDevices, int numPoints,
                  int swpCnt, int resultFormat, int segSwp, bool avgSwpData) {
  FILE *fp = fopen(csvPath, "w");
  if (!fp) {
    fprintf(stderr, "[ERROR] Cannot open %s for writing\n", csvPath);
    return;
  }
  if (segSwp == 1) {
    if (avgSwpData) {
      PrintSweepResToFileSegmented(fp, numOfDevices, numPoints, 0,
                                   resultFormat, SParam1shm, SParam2shm);
    } else {
      for (int i = 0; i < swpCnt; i++) {
        PrintSweepResToFileSegmented(fp, numOfDevices, numPoints, i,
                                     resultFormat, SParam1shm, SParam2shm);
      }
    }
  } else {
    if (avgSwpData) {
      PrintSweepResToFile(fp, numOfDevices, numPoints, 0,
                          resultFormat, SParam1shm, SParam2shm);
    } else {
      for (int i = 0; i < swpCnt; i++) {
        PrintSweepResToFile(fp, numOfDevices, numPoints, i,
                            resultFormat, SParam1shm, SParam2shm);
      }
    }
  }
  fclose(fp);

  // Post-process: keep only S-parameters for the antennas the user said
  // they're using. The Keysight library always writes the full 4-port matrix
  // (Frequency + 16 mag/phase pairs); this strips out columns for ports
  // higher than g_bs_numAntennas.
  if (g_bs_numAntennas >= 1 && g_bs_numAntennas < 4) {
    bs_filter_csv_to_antennas(csvPath, g_bs_numAntennas);
  }
}

// Filter a freshly-written 4-port CSV down to only the S-parameters that
// involve ports 1..numAntennas. Columns kept:
//   * Frequency (column 0, always)
//   * For each Sij pair where i <= numAntennas AND j <= numAntennas: both
//     the magnitude and phase columns
//
// Column layout in the original CSV (1-indexed pairs, 0-indexed cols):
//   col 0   : Frequency
//   cols 1-2: S1-1 mag, S1-1 phase   (pair_idx 0,  i=1, j=1)
//   cols 3-4: S2-1 mag, S2-1 phase   (pair_idx 1,  i=2, j=1)
//   cols 5-6: S3-1 mag, S3-1 phase   (pair_idx 2,  i=3, j=1)
//   cols 7-8: S4-1 mag, S4-1 phase   (pair_idx 3,  i=4, j=1)
//   cols 9-10: S1-2 ...              (pair_idx 4,  i=1, j=2)
//   ... and so on through col 31-32 = S4-4
//
// So for a column c >= 1: pair_idx = (c-1)/2, i = pair_idx%4 + 1, j = pair_idx/4 + 1.
// Keep the column if both i and j are <= numAntennas.
void bs_filter_csv_to_antennas(const char *csvPath, int numAntennas) {
  if (numAntennas < 1 || numAntennas >= 4) return;  // no-op

  FILE *fp = fopen(csvPath, "r");
  if (!fp) return;

  // Read the entire file into memory.
  fseek(fp, 0, SEEK_END);
  long fsize = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  if (fsize <= 0) { fclose(fp); return; }

  char *content = (char*)malloc((size_t)fsize + 1);
  if (!content) { fclose(fp); return; }
  size_t nread = fread(content, 1, (size_t)fsize, fp);
  content[nread] = '\0';
  fclose(fp);

  // Re-open for writing (truncate).
  fp = fopen(csvPath, "w");
  if (!fp) { free(content); return; }

  // Walk through the buffer one line at a time.
  char *line_start = content;
  char *buf_end = content + nread;
  while (line_start < buf_end) {
    // Find end of this line.
    char *line_end = line_start;
    while (line_end < buf_end && *line_end != '\n') line_end++;

    // Empty line: just write the newline if present and move on.
    if (line_end == line_start) {
      if (line_end < buf_end) fputc('\n', fp);
      line_start = line_end + 1;
      continue;
    }

    // Walk through comma-separated tokens in [line_start, line_end).
    char *tok_start = line_start;
    int col = 0;
    int first_kept = 1;
    while (tok_start <= line_end) {
      char *tok_end = tok_start;
      while (tok_end < line_end && *tok_end != ',') tok_end++;

      // Decide whether to keep this column.
      int keep = 0;
      if (col == 0) {
        keep = 1;
      } else {
        int pair_idx = (col - 1) / 2;
        if (pair_idx >= 0 && pair_idx < 16) {
          int i = pair_idx % 4 + 1;
          int j = pair_idx / 4 + 1;
          if (i <= numAntennas && j <= numAntennas) keep = 1;
        }
      }

      // Don't write trailing empty token (from the trailing comma at end-of-line)
      // unless col 0. The CSVs end with a "...,P4-4," with a trailing comma.
      int is_trailing_empty = (tok_start == tok_end) && (col > 0) && (tok_end == line_end);

      if (keep && !is_trailing_empty) {
        if (!first_kept) fputc(',', fp);
        fwrite(tok_start, 1, (size_t)(tok_end - tok_start), fp);
        first_kept = 0;
      }

      if (tok_end >= line_end) break;
      tok_start = tok_end + 1;
      col++;
    }

    fputc('\n', fp);
    line_start = line_end + 1;
  }

  free(content);
  fclose(fp);
}

// ---------- Persistent config helpers ----------

int bs_load_list(const char *filename, char items[][BS_NAME_LEN], int maxItems) {
  char path[512];
  snprintf(path, sizeof(path), "%s/%s", BS_CONFIG_DIR, filename);
  FILE *f = fopen(path, "r");
  if (!f) return 0;
  int n = 0;
  char buf[BS_NAME_LEN];
  while (n < maxItems && fgets(buf, sizeof(buf), f)) {
    size_t L = strlen(buf);
    while (L > 0 && (buf[L-1] == '\n' || buf[L-1] == '\r')) buf[--L] = '\0';
    if (L == 0) continue;
    strncpy(items[n], buf, BS_NAME_LEN - 1);
    items[n][BS_NAME_LEN - 1] = '\0';
    n++;
  }
  fclose(f);
  return n;
}

void bs_save_list(const char *filename, char items[][BS_NAME_LEN], int count) {
  char path[512];
  snprintf(path, sizeof(path), "%s/%s", BS_CONFIG_DIR, filename);
  FILE *f = fopen(path, "w");
  if (!f) return;
  for (int i = 0; i < count; i++) fprintf(f, "%s\n", items[i]);
  fclose(f);
}

int bs_append_unique(const char *filename, const char *item) {
  if (item[0] == '\0') return 0;
  char items[BS_MAX_SAVED][BS_NAME_LEN];
  int n = bs_load_list(filename, items, BS_MAX_SAVED);
  for (int i = 0; i < n; i++) {
    if (strcmp(items[i], item) == 0) return 0;  // already there
  }
  if (n >= BS_MAX_SAVED) return 0;
  strncpy(items[n], item, BS_NAME_LEN - 1);
  items[n][BS_NAME_LEN - 1] = '\0';
  n++;
  bs_save_list(filename, items, n);
  return 1;
}

int bs_load_grid_config(const char *modelName, int *gr, int *gc,
                        int *measureRows, int *nMR, int *measureCols, int *nMC,
                        double *cellSize, double *divider) {
  char cleanName[BS_NAME_LEN];
  bs_sanitize(modelName, cleanName, sizeof(cleanName));
  char path[512];
  snprintf(path, sizeof(path), "%s/model_%s.conf", BS_CONFIG_DIR, cleanName);
  FILE *f = fopen(path, "r");
  if (!f) return 0;
  char buf[512];
  *nMR = 0; *nMC = 0;
  while (fgets(buf, sizeof(buf), f)) {
    if (strncmp(buf, "GridRows=", 9) == 0) *gr = atoi(buf + 9);
    else if (strncmp(buf, "GridCols=", 9) == 0) *gc = atoi(buf + 9);
    else if (strncmp(buf, "CellSize=", 9) == 0) *cellSize = atof(buf + 9);
    else if (strncmp(buf, "Divider=", 8) == 0) *divider = atof(buf + 8);
    else if (strncmp(buf, "MeasureRows=", 12) == 0) {
      char *tok = strtok(buf + 12, " \n\r\t,");
      while (tok && *nMR < 64) { measureRows[(*nMR)++] = atoi(tok); tok = strtok(NULL, " \n\r\t,"); }
    }
    else if (strncmp(buf, "MeasureCols=", 12) == 0) {
      char *tok = strtok(buf + 12, " \n\r\t,");
      while (tok && *nMC < 64) { measureCols[(*nMC)++] = atoi(tok); tok = strtok(NULL, " \n\r\t,"); }
    }
  }
  fclose(f);
  return (*nMR > 0 && *nMC > 0);
}

void bs_save_grid_config(const char *modelName, int gr, int gc,
                         int *measureRows, int nMR, int *measureCols, int nMC,
                         double cellSize, double divider) {
  char cleanName[BS_NAME_LEN];
  bs_sanitize(modelName, cleanName, sizeof(cleanName));
  char path[512];
  snprintf(path, sizeof(path), "%s/model_%s.conf", BS_CONFIG_DIR, cleanName);
  FILE *f = fopen(path, "w");
  if (!f) return;
  fprintf(f, "ModelName=%s\n", modelName);
  fprintf(f, "GridRows=%d\n", gr);
  fprintf(f, "GridCols=%d\n", gc);
  fprintf(f, "MeasureRows=");
  for (int i = 0; i < nMR; i++) fprintf(f, "%d%s", measureRows[i], (i < nMR - 1) ? " " : "");
  fprintf(f, "\nMeasureCols=");
  for (int i = 0; i < nMC; i++) fprintf(f, "%d%s", measureCols[i], (i < nMC - 1) ? " " : "");
  fprintf(f, "\nCellSize=%.6f\nDivider=%.6f\n", cellSize, divider);
  fclose(f);
}

int bs_load_skip_list(const char *modelName, char skips[][BS_LABEL_LEN], int maxSkips) {
  char cleanName[BS_NAME_LEN];
  bs_sanitize(modelName, cleanName, sizeof(cleanName));
  char path[512];
  snprintf(path, sizeof(path), "%s/skips_%s.list", BS_CONFIG_DIR, cleanName);
  FILE *f = fopen(path, "r");
  if (!f) return 0;
  int n = 0;
  char buf[BS_LABEL_LEN + 8];
  while (n < maxSkips && fgets(buf, sizeof(buf), f)) {
    size_t L = strlen(buf);
    while (L > 0 && (buf[L-1] == '\n' || buf[L-1] == '\r')) buf[--L] = '\0';
    if (L == 0) continue;
    strncpy(skips[n], buf, BS_LABEL_LEN - 1);
    skips[n][BS_LABEL_LEN - 1] = '\0';
    n++;
  }
  fclose(f);
  return n;
}

void bs_save_skip_list(const char *modelName, char skips[][BS_LABEL_LEN], int count) {
  char cleanName[BS_NAME_LEN];
  bs_sanitize(modelName, cleanName, sizeof(cleanName));
  char path[512];
  snprintf(path, sizeof(path), "%s/skips_%s.list", BS_CONFIG_DIR, cleanName);
  FILE *f = fopen(path, "w");
  if (!f) return;
  for (int i = 0; i < count; i++) fprintf(f, "%s\n", skips[i]);
  fclose(f);
}

void bs_merge_and_save_skips(const char *modelName,
                             char existingSkips[][BS_LABEL_LEN], int nExisting,
                             char newSkips[][BS_LABEL_LEN], int nNew) {
  char merged[BS_MAX_SKIPS][BS_LABEL_LEN];
  int nMerged = 0;
  // copy existing
  for (int i = 0; i < nExisting && nMerged < BS_MAX_SKIPS; i++) {
    strncpy(merged[nMerged], existingSkips[i], BS_LABEL_LEN - 1);
    merged[nMerged][BS_LABEL_LEN - 1] = '\0';
    nMerged++;
  }
  // append new ones if not already present
  for (int i = 0; i < nNew; i++) {
    int dup = 0;
    for (int j = 0; j < nMerged; j++) {
      if (strcmp(merged[j], newSkips[i]) == 0) { dup = 1; break; }
    }
    if (!dup && nMerged < BS_MAX_SKIPS) {
      strncpy(merged[nMerged], newSkips[i], BS_LABEL_LEN - 1);
      merged[nMerged][BS_LABEL_LEN - 1] = '\0';
      nMerged++;
    }
  }
  bs_save_skip_list(modelName, merged, nMerged);
}

int bs_is_in_skip_list(const char *label, char skips[][BS_LABEL_LEN], int n) {
  for (int i = 0; i < n; i++) {
    if (strcmp(skips[i], label) == 0) return 1;
  }
  return 0;
}

int bs_menu_pick(const char *category,
                 const char **builtins, int nBuiltins,
                 char saved[][BS_NAME_LEN], int nSaved,
                 char *result, size_t resultSize, int *wasNew) {
  *wasNew = 0;
  while (1) {
    printf("\n%s:\n", category);
    int idx = 1;
    for (int i = 0; i < nBuiltins; i++) printf("  [%2d] %s\n", idx++, builtins[i]);
    for (int i = 0; i < nSaved; i++)   printf("  [%2d] %s (saved)\n", idx++, saved[i]);
    int newIdx = idx;
    printf("  [%2d] (enter new)\n", newIdx);
    int total = nBuiltins + nSaved + 1;

    char buf[64];
    char prompt[128];
    snprintf(prompt, sizeof(prompt), "Select [1]: ");
    bs_read_line(prompt, buf, sizeof(buf));
    int choice = (buf[0] == '\0') ? 1 : atoi(buf);
    if (choice < 1 || choice > total) {
      printf("  Invalid choice, try again.\n");
      continue;
    }
    if (choice <= nBuiltins) {
      strncpy(result, builtins[choice - 1], resultSize - 1);
      result[resultSize - 1] = '\0';
      return 0;
    }
    if (choice <= nBuiltins + nSaved) {
      strncpy(result, saved[choice - nBuiltins - 1], resultSize - 1);
      result[resultSize - 1] = '\0';
      return 0;
    }
    // (enter new)
    char nameBuf[BS_NAME_LEN];
    bs_read_line("Enter new name: ", nameBuf, sizeof(nameBuf));
    if (nameBuf[0] == '\0') {
      printf("  Empty name, try again.\n");
      continue;
    }
    strncpy(result, nameBuf, resultSize - 1);
    result[resultSize - 1] = '\0';
    *wasNew = 1;
    return 0;
  }
}
// =============================================================
// end BATCH SWEEP v2 helpers
// =============================================================

