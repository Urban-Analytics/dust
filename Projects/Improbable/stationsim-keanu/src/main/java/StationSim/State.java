package StationSim;

public class State {

    private static final long serialVersionUID = 1;

    public long step;
    public String person;
    public int finished;
    public double x;
    public double y;
    public double speed;
    public String entrance;
    public String exit;

    public State(){}

    public State(long step, String person, int finished, double x, double y,
                 double speed, String entrance, String exit) {
        this.step = step;
        this.person = person;
        this.finished = finished;
        this.x = x;
        this.y = y;
        this.speed = speed;
        this.entrance = entrance;
        this.exit = exit;
    }

    @Override
    public String toString() {
        return(Long.toString(step) + ", "
                + person + ", "
                + finished + ", "
                + x + ", "
                + y + ", "
                + speed + ", "
                + exit);
    }
}

