package examples.exercise_pacman;

import pacman.controllers.PacmanController;
import pacman.game.Constants;
import pacman.game.Constants.MOVE;
import pacman.game.Game;
import java.util.ArrayList;
import java.util.Random;


public class MyPacMan extends PacmanController {
    //Min distance to start run away from ghost
    private static final int MIN_DISTANCE = 50;
    //The time MsPacMan ate last power pill
    int timePowerpill = Integer.MAX_VALUE;
    private Random random = new Random();
    //In target_pills we store available pills position inorder to eat them
    public   ArrayList<Integer> target_pills = new ArrayList<>();
    //Here we store last ghost known position and the time we saw it
    private Integer [][] ghosts = new Integer [2][4];
    //Here we store last edible ghost known position
    private Integer [] ghostsEdible = new Integer [4];
    //General memory for ghosts positions
    private Integer [] ghostGeneral = new Integer [4];
    //Misc variable
    private Constants.GHOST[] ghostEntity = new Constants.GHOST[4];
    //Power pills known positions
    public ArrayList<Integer> power_pills = new ArrayList<>();


//The idea behind our MsPacMan was similar to a DTree: we check some condition and if they fail we go to the next behaviour
public MOVE getMove(Game game, long timeDue) {

        //Update ghosts location if you see them
        ghost_location(game);

        //Misc variable for pills potision
        ArrayList<Integer> target ;
        target = updateTargets(game);

        //If game is at the begining reset power pills position
        if (game.getCurrentLevelTime()<10){
            target_pills = new ArrayList<>();
            timePowerpill = Integer.MAX_VALUE;
        }

        //If a ghost is edible we hunt the ghost
        MOVE strategy1 = ghostRun(game);
        //If return null just go to the next behaviour
        if (strategy1!=null){
            return strategy1;
        }
        //If you see ghost run away
        MOVE strategy2 = catchTheGhost(game);
        //If return null just go to the next behaviour
        if(strategy2!=null){
            return strategy2;
        }
        //If the game is at the begining do random moves inorder to explore
        if (game.getCurrentLevelTime()<100){
            return randomMove(game);
        }
        //Go to the closest power pill known position
        MOVE strategy3 = pillHunting(game,target);
        //If return null just go to the next behaviour
        if(strategy3!=null){
            return strategy3;
        }
        //Do random move
        MOVE strategy4 = randomMove(game);
        //If return null just go to the next behaviour
        if(strategy4!=null){
            return strategy4;
        }

        //If all other fail just turn around
        return game.getPacmanLastMoveMade().opposite();
    }

//Here we find the closest Ghost and run away from it
public MOVE ghostRun(Game game) {

    MOVE no_ghost =null;
    int minDistance =MIN_DISTANCE;
    int closestGhost=-1;
    int current = game.getPacmanCurrentNodeIndex();
    ghost_location(game);

    for (int i =0 ; i<4 ; i++) {

            if (ghosts[0][i] != null) {
                int distance = game.getShortestPathDistance(current,ghosts[0][i]);
                if (distance < minDistance) {
                    minDistance=distance;
                    closestGhost=i;
            }
        }
    }
        if (closestGhost!=-1){

            return game.getNextMoveAwayFromTarget(current, ghosts[0][closestGhost] ,Constants.DM.PATH );
        }
        //If there are no ghosts near MsPacMan just return null
      return no_ghost ;
    }

//Here we find the closest edible ghost and hunt it
public MOVE catchTheGhost(Game game){

    int current = game.getPacmanCurrentNodeIndex();
    MOVE catch_ghost =null;
    int minDistance = Integer.MAX_VALUE;
    int ghostNumber=-1;
    boolean randMove=false;
    ghost_location(game);

    for (int i =0 ;i<ghostsEdible.length ;i++) {
        if (ghostsEdible[i]!=null) {
            randMove=true;
            int distance = game.getShortestPathDistance(current, ghostsEdible[i]);

            if ((game.getCurrentLevelTime() - timePowerpill)-distance > -50){

                if (distance < minDistance) {
                    minDistance = distance;
                    ghostNumber=i;
                }
            }

        }
    }
    if (ghostNumber!=-1) {
        return game.getNextMoveTowardsTarget(current, ghostsEdible[ghostNumber], Constants.DM.PATH);
    }else if (randMove){
        return randomMove(game);
    }
      //If there is no edible ghost return null
      return catch_ghost;
}

//Here we find the closest pill and go to it
public MOVE pillHunting (Game game, ArrayList<Integer> target){

    MOVE pillsHunt =null;
    int current = game.getPacmanCurrentNodeIndex();
    if (!target.isEmpty()) {
        int[] targetsArray = new int[target.size()];        //convert from ArrayList to array

        for (int i = 0; i < targetsArray.length; i++) {
            targetsArray[i] = target.get(i);
        }
        MOVE move = game.getNextMoveTowardsTarget(current, game.getClosestNodeIndexFromNodeIndex(current, targetsArray, Constants.DM.PATH), Constants.DM.PATH);

        return move;
    }
    //If there are no pills return null
    return pillsHunt;
}

//Here we move random in one of available positions
public MOVE randomMove (Game game){

    MOVE randomMove = null;

    int current = game.getPacmanCurrentNodeIndex();
    MOVE[] moves = game.getPossibleMoves(current, game.getPacmanLastMoveMade());
    if (moves.length > 0) {
        return moves[random.nextInt(moves.length)];
    }
        //If you can't do random move return null
        return randomMove;
}

//Here we update pills position, if eaten remove it from available list
public ArrayList<Integer> updateTargets (Game game){

    int[] pills = game.getPillIndices();
    for (int i = 0; i < pills.length; i++) {
        //check which pills are available
        Boolean pillStillAvailable = game.isPillStillAvailable(i);
        if (pillStillAvailable != null) {
            if (pillStillAvailable && !target_pills.contains(pills[i])) {
                target_pills.add(pills[i]);
            }
            else if (target_pills.contains(pills[i]) && !pillStillAvailable) {
                target_pills.remove(new Integer(pills[i]));
            }
        }

    }
    return target_pills;
}

//Another behaviour for haunting the pill, not used!!
public MOVE pillHunting2(Game game,ArrayList<Integer> targets) {

        int current = game.getPacmanCurrentNodeIndex();
        MOVE pillsHunt =null;
        int[] powerPills = game.getPowerPillIndices();

        for (int i = 0; i < powerPills.length; i++) {            //check with power pills are available
            Boolean pillStillAvailable = game.isPillStillAvailable(i);
            if (pillStillAvailable != null) {
                if (pillStillAvailable) {
                    targets.add(powerPills[i]);
                }
            }
        }
        if (!targets.isEmpty()) {
            int[] targetsArray = new int[targets.size()];        //convert from ArrayList to array

            for (int i = 0; i < targetsArray.length; i++) {
                targetsArray[i] = targets.get(i);
            }
            return game.getNextMoveTowardsTarget(current, game.getClosestNodeIndexFromNodeIndex(current, targetsArray, Constants.DM.PATH), Constants.DM.PATH);
        }

    return  pillsHunt;
    }

//Here we update ghosts location and time, if we see them
public void ghost_location(Game game){

        int current = game.getPacmanCurrentNodeIndex();
        int i=0;
        for (Constants.GHOST ghost : Constants.GHOST.values()) {
            ghostEntity[i] = ghost;
            int ghostLocation = game.getGhostCurrentNodeIndex(ghostEntity[i]);
            //If it's not edible store it to ghosts
            if(ghostLocation!=-1){
                ghostGeneral[i]=ghostLocation;
            }
            //If ghost was eaten remove it from memory
            if(game.wasGhostEaten(ghost)){
                ghostGeneral[i]=null;
                ghostsEdible[i] = null;
            }
            //If MsPacman is at the last known ghost position remove it from memory
            if(ghostsEdible[i]!=null){
                if(ghostsEdible[i]==game.getPacmanCurrentNodeIndex()){
                    ghostsEdible[i]=null;
                    ghostGeneral[i]=null;
                }
            }
            //If you see ghost save its position and the time you saw it
            if(ghostLocation!=-1 && !(game.getGhostEdibleTime(ghost)>0)){
                ghosts[0][i] = ghostLocation;
                ghosts[1][i] = game.getCurrentLevelTime();
            }
            //If it is edible store it to ghostsEdible
            if(ghostLocation!=-1 && game.getGhostEdibleTime(ghost) > 0){
                ghostsEdible[i] = ghostLocation;

            }
            //If ghost is not edible anymore, remove it from ghostsEdible
            if (!(game.getGhostEdibleTime(ghost) > 0)){
                //ghostsEdible[i] = null;
            }

            //If ghost is far away, remove it from ghosts
            if (ghosts[0][i] !=null){
                int distance = game.getShortestPathDistance(current,ghosts[0][i]);
                if (distance>70){
                    ghosts[0][i]=null;
                    ghosts[1][i]=null;
                }
            }
            //If there is a lot of time past since you last saw ghost, remove it's position
            if (ghosts[1][i] !=null){
                if (game.getCurrentLevelTime()-ghosts[1][i] >15){
                    ghosts[0][i]=null;
                    ghosts[1][i]=null;
                }
            }
            i=i+1;
                }
    //If you eat power pill fill the ghostsEdible array and empty the ghosts
    if (game.wasPowerPillEaten()) {
        timePowerpill = game.getCurrentLevelTime();

        if (game.getCurrentLevelTime() - timePowerpill < 200) {
            for (int j = 0; j < ghostGeneral.length; j++) {
                ghostsEdible[j] = ghostGeneral[j];
                ghosts[0][j]=null;
                ghosts[1][j]=null;

            }
        }
    }
    //If the power pill time end restore your memory
    if(game.getCurrentLevelTime()-timePowerpill >200){

        for (int j=0;j<ghostGeneral.length;j++){
            ghostsEdible[j]=null;
        }
    }


    }

//Here we implement a behaviour for going to power pill, not used!!
public MOVE gotoPowerPill(Game game){


        int[] powerPills = game.getPowerPillIndices();
        int current = game.getPacmanCurrentNodeIndex();

        for (int i = 0; i < powerPills.length; i++) {            //check with power pills are available
            Boolean pillStillAvailable = game.isPillStillAvailable(i);
            if (pillStillAvailable != null) {
                if (pillStillAvailable) {
                    power_pills.add(powerPills[i]);
                } else {

                    power_pills.remove(new Integer(powerPills[i]));

                }
            }
        }

        int[] powerPillsArray = new int[power_pills.size()];        //convert from ArrayList to array

        for (int i = 0; i < powerPillsArray.length; i++) {
            powerPillsArray[i] = power_pills.get(i);

        }



        MOVE move = game.getNextMoveTowardsTarget(current, game.getClosestNodeIndexFromNodeIndex(current, powerPillsArray, Constants.DM.PATH), Constants.DM.PATH);
        return move;
    }

//Check if powerPill still available
public boolean powerPillAvailable(Game game){

        int[] powerPills = game.getPowerPillIndices();
        boolean isAvailable = false;

        for (int i = 0; i < powerPills.length; i++) {
            Boolean pillStillAvailable = game.isPillStillAvailable(i);
            if (pillStillAvailable != null) {
                if (pillStillAvailable) {
                    isAvailable=true;
                    System.out.println(i);
                }
            }
        }
        return  isAvailable;
    }
}

