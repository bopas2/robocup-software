public class Catch {

    public static void main(String[] args) {
        try {
            v();
        } catch (Exception e) {
            System.out.println("B");
        }
    }

    static public void v() {
        try {
            throw new Exception();
        } catch(Exception e) {
            System.out.println("A");
        }
    }
}