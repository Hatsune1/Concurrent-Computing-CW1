// COMS20001 - Cellular Automaton Farm - Initial Code Skeleton
// (using the XMOS i2c accelerometer demo code)

#include <platform.h>
#include <xs1.h>
#include <stdio.h>
#include "pgmIO.h"
#include "i2c.h"

#define  IMHT 16                  //image height
#define  IMWD 16                  //image width
#define  DEAD 0
#define  ALIVE 255
#define  SEP_GREEN_LED 1
#define  BLUE_LED 2
#define  GREEN_LED 4
#define  RED_LED 8

typedef unsigned char uchar;      //using uchar as shorthand

port p_scl = XS1_PORT_1E;         //interface ports to orientation
port p_sda = XS1_PORT_1F;

#define FXOS8700EQ_I2C_ADDR 0x1E  //register addresses for orientation
#define FXOS8700EQ_XYZ_DATA_CFG_REG 0x0E
#define FXOS8700EQ_CTRL_REG_1 0x2A
#define FXOS8700EQ_DR_STATUS 0x0
#define FXOS8700EQ_OUT_X_MSB 0x1
#define FXOS8700EQ_OUT_X_LSB 0x2
#define FXOS8700EQ_OUT_Y_MSB 0x3
#define FXOS8700EQ_OUT_Y_LSB 0x4
#define FXOS8700EQ_OUT_Z_MSB 0x5
#define FXOS8700EQ_OUT_Z_LSB 0x6

on tile[0] : in port buttons = XS1_PORT_4E; //port to access xCore-200 buttons
on tile[0] : out port leds = XS1_PORT_4F;   //port to access xCore-200 LEDs

int showLEDs(out port p, chanend fromDist) {
  int pattern;
  while (1) {
    fromDist :> pattern;   //receive new pattern from visualiser
    if (pattern == -1) {
        p <: 0;
        break;
    }
    else p <: pattern;                //send pattern to LED port
  }
  return 0;
}

//READ BUTTONS and send button pattern to userAnt
void buttonListener(in port b, chanend inToDist, chanend outToDist) {
  int r;
  while (1) {
    b when pinseq(15)  :> r;                 // check that no button is pressed
    b when pinsneq(15) :> r;                 // check if some buttons are pressed
    if (r==13) inToDist <: r;                // send button pattern to userAnt
    if (r==14) outToDist <: r;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////
//
// Read Image from PGM file from path infname[] to channel c_out
//
/////////////////////////////////////////////////////////////////////////////////////////
void DataInStream(char infname[], chanend c_out)
{
  int res;
  uchar line[ IMWD ];

  printf( "DataInStream: Start...\n" );

  //Open PGM file
  res = _openinpgm( infname, IMWD, IMHT );
  if( res ) {
    printf( "DataInStream: Error openening %s\n.", infname );
    return;
  }

  //Read image line-by-line and send byte by byte to channel c_out
  for( int y = 0; y < IMHT; y++ ) {
    _readinline( line, IMWD );
    for( int x = 0; x < IMWD; x++ ) {
      c_out <: line[ x ];
    }
  }

  //Close PGM image file
  _closeinpgm();
  printf( "DataInStream: Done...\n" );
  return;
}

int ToLiveOrNotToLive (uchar image[height][IMWD], int y, int x, unsigned height) {
    int dead = 0;
    int alive = 0;

    if (image[(y-1+height)%height][(x-1+IMWD)%IMWD] == DEAD) dead += 1;
    else if (image[(y-1+height)%height][(x-1+IMWD)%IMWD] == ALIVE) alive += 1;

    if (image[(y-1+height)%height][x] == DEAD) dead += 1;
    else if (image[(y-1+height)%height][x] == ALIVE) alive += 1;

    if (image[(y-1+height)%height][(x+1+IMWD)%IMWD] == DEAD) dead += 1;
    else if (image[(y-1+height)%height][(x+1+IMWD)%IMWD] == ALIVE) alive += 1;

    if (image[y][(x-1+IMWD)%IMWD] == DEAD) dead += 1;
    else if (image[y][(x-1+IMWD)%IMWD] == ALIVE) alive += 1;

    if (image[y][(x+1+IMWD)%IMWD] == DEAD) dead += 1;
    else if (image[y][(x+1+IMWD)%IMWD] == ALIVE) alive += 1;

    if (image[(y+1+height)%height][(x-1+IMWD)%IMWD] == DEAD) dead += 1;
    else if (image[(y+1+height)%height][(x-1+IMWD)%IMWD] == ALIVE) alive += 1;

    if (image[(y+1+height)%height][x] == DEAD) dead += 1;
    else if (image[(y+1+height)%height][x] == ALIVE) alive += 1;

    if (image[(y+1+height)%height][(x+1+IMWD)%IMWD] == DEAD) dead += 1;
    else if (image[(y+1+height)%height][(x+1+IMWD)%IMWD] == ALIVE) alive += 1;

    if (alive < 2 && image[y][x] == ALIVE) return DEAD;
    else if (alive > 3 && image[y][x] == ALIVE) return DEAD;
    else if (alive == 3 && image[y][x] == DEAD) return ALIVE;
    else return image[y][x];
}

void distributor (chanend fromIN, chanend toOUT, chanend fromORI,
        chanend fromButtonIN, chanend fromButtonOUT, chanend toLED
        /*chanend worker1, chanend work2, chanend worker3, chanend worker4*/){
    int value;
    int round = 0;
    int liveNum = 0;

    timer tmr;
    int startTime;
    int stopTime;
    int totalTime;
    int pauseTime = 0;
    int isPaused = 0;

    uchar val;
    uchar image [IMHT][IMWD];
    uchar result [IMHT][IMWD];

    printf( "ProcessImage: Start, size = %dx%d\n", IMHT, IMWD );
    printf( "Waiting for Button Press...\n" );
    fromButtonIN :> value;
    printf( "Processing...\n" );

    // Show green LED
    toLED <: GREEN_LED;
    // When the Read button was pressed, start to read original data from file.
    for( int y = 0; y < IMHT; y++ ) {
        for( int x = 0; x < IMWD; x++ ) {
            fromIN :> val;
            image[y][x] = val;
        }
    }
    // Count initial live number.
    for( int y = 0; y < IMHT; y++ ) {
        for( int x = 0; x < IMWD; x++ ) {
            if (image[y][x] == ALIVE) liveNum ++;
        }
    }

    // Record start time.
    tmr :> startTime;
    // Execute the main loop.
    while(1){
        select {
            // When the output button was pressed, export data and show blue LED
            case fromButtonOUT :> value:{
                toLED <: BLUE_LED;
                for( int y = 0; y < IMHT; y++ ) {
                    for( int x = 0; x < IMWD; x++ ) {
                        toOUT <: image[y][x];
                    }
                }
                break;
            }
            // When tilted, report the number of rounds and live cells,
            // time passed after read file, and show red LED.
            case fromORI :> value:{
                // Record stop time and report only when tilt begin.
                if(isPaused == 0){
                    tmr :> stopTime;
                    totalTime = stopTime - startTime - pauseTime;
                    printf("Round: %d\n", round);
                    printf("Live number is %d\n", liveNum);
                    int printTimeSec = totalTime / 100000000;
                    int printTimeMiliSec = (totalTime - printTimeSec * 100000000) / 100000;
                    printf("Time passed: %d.%ds\n", printTimeSec, printTimeMiliSec);
                    isPaused = 1;
                }
                toLED <: RED_LED;
                break;
            }
            default:{
                // If just let the board down, calculate pause time and set bool to 0.
                if (isPaused == 1){
                    tmr :> value;
                    pauseTime += value - stopTime;
                    isPaused = 0;
                }
                // Convert image to result for one time.
                for( int y = 0; y < IMHT; y++ ) {
                    for( int x = 0; x < IMWD; x++ ) {
                        result [y][x] = ToLiveOrNotToLive (image, y, x, IMHT);
                        //printf( "-%4.1d ", result[y][x] );
                    }
                    //printf("\n");
                }
                //printf( "\nOne processing round completed...\n" );
                //Pass processed image to next round
                for( int y = 0; y < IMHT; y++ ) {
                    for( int x = 0; x < IMWD; x++ ) {
                        image[y][x] = result[y][x];
                    }
                }
                liveNum = 0;
                for( int y = 0; y < IMHT; y++ ) {
                    for( int x = 0; x < IMWD; x++ ) {
                        if (image[y][x] == ALIVE) liveNum ++;
                    }
                }
                // Add 1 to round and show separate green LED.
                round += 1;
                toLED <: SEP_GREEN_LED;
                break;
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////
//
// Write pixel stream from channel c_in to PGM image file
//
/////////////////////////////////////////////////////////////////////////////////////////
void DataOutStream(char outfname[], chanend c_in)
{
  int res;
  uchar line[ IMWD ];

  //Open PGM file
  printf( "DataOutStream: Start...\n" );
  res = _openoutpgm( outfname, IMWD, IMHT );
  if( res ) {
    printf( "DataOutStream: Error opening %s\n.", outfname );
    return;
  }

  //Compile each line of the image and write the image line-by-line
  while(1) {
      for( int y = 0; y < IMHT; y++ ) {
          for( int x = 0; x < IMWD; x++ ) {
              c_in :> line[ x ];
          }
          _writeoutline( line, IMWD );
          printf( "DataOutStream: Line written...\n" );
      }

      //Close the PGM image
      _closeoutpgm();
      printf( "DataOutStream: Done...\n" );
  }
}

/////////////////////////////////////////////////////////////////////////////////////////
//
// Initialise and  read orientation, send first tilt event to channel
//
/////////////////////////////////////////////////////////////////////////////////////////
void orientation( client interface i2c_master_if i2c, chanend toDist) {
  i2c_regop_res_t result;
  char status_data = 0;

  // Configure FXOS8700EQ
  result = i2c.write_reg(FXOS8700EQ_I2C_ADDR, FXOS8700EQ_XYZ_DATA_CFG_REG, 0x01);
  if (result != I2C_REGOP_SUCCESS) {
    printf("I2C write reg failed\n");
  }
  
  // Enable FXOS8700EQ
  result = i2c.write_reg(FXOS8700EQ_I2C_ADDR, FXOS8700EQ_CTRL_REG_1, 0x01);
  if (result != I2C_REGOP_SUCCESS) {
    printf("I2C write reg failed\n");
  }

  //Probe the orientation x-axis forever
  while (1) {

    //check until new orientation data is available
    do {
      status_data = i2c.read_reg(FXOS8700EQ_I2C_ADDR, FXOS8700EQ_DR_STATUS, result);
    } while (!status_data & 0x08);

    //get new x-axis tilt value
    int x = read_acceleration(i2c, FXOS8700EQ_OUT_X_MSB);

    //send signal to distributor if tilted
    if (x>30) toDist <: 1;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////
//
// Orchestrate concurrent system and start up all threads
//
/////////////////////////////////////////////////////////////////////////////////////////
int main(void) {

i2c_master_if i2c[1];                                 //interface to orientation

char infname[] = "test.pgm";                          //put your input image path here
char outfname[] = "testout.pgm";                      //put your output image path here
chan c_inIO, c_outIO,
     orientationToDistributor,
     button1ToDistributor, button2ToDistributor,
     distributorToLEDs;                               //extend your channel definitions here

par {
    i2c_master(i2c, 1, p_scl, p_sda, 10);             //server thread providing orientation data
    orientation(i2c[0],orientationToDistributor);     //client thread reading orientation data
    DataInStream(infname, c_inIO);                    //thread to read in a PGM image
    DataOutStream(outfname, c_outIO);                 //thread to write out a PGM image
    distributor(c_inIO, c_outIO, orientationToDistributor, button1ToDistributor, button2ToDistributor, distributorToLEDs);
    buttonListener(buttons, button1ToDistributor, button2ToDistributor);
    showLEDs(leds,distributorToLEDs);
  }

  return 0;
}
