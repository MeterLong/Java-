public class Test_1 {

    public static void main(String[] args) {
        try {
            System.out.println("******************研究生一*********************");
            Doctor xm = new Doctor();
            xm.setName("王小明");
            xm.setAge(20);
            xm.setNumber(2019666888);
            xm.setSex("男");
            xm.setTuition(8000);
            xm.setSalary(1200);
            System.out.println("学生姓名:" + xm.getName());
            System.out.println("学生年龄:" + xm.getAge());
            System.out.println("学生编号:" + xm.getNumber());
            System.out.println("学生性别:" + xm.getSex());
            xm.find_tuition();
            xm.find_salary();
            xm.taxation();
            System.out.println("******************研究生二*********************");
            Doctor xf = new Doctor();
            xf.setName("董晓芳");
            xf.setAge(20);
            xf.setNumber(2019666999);
            xf.setSex("女");
            xf.setTuition(8000);
            xf.setSalary(1185);
            System.out.println("学生姓名:" + xf.getName());
            System.out.println("学生年龄:" + xf.getAge());
            System.out.println("学生编号:" + xf.getNumber());
            System.out.println("学生性别:" + xf.getSex());
            xf.find_tuition();
            xf.find_salary();
            xf.taxation();
        } catch (Exception e) {
            System.out.println("数据异常");
        }

    }

    interface Manger_student {
        double find_tuition();

        double afford_tuition();
    }

    interface Manger_teacher {
        double STANDARD = 0.2;

        double find_salary();

        double get_salary();
    }


    public static class Doctor implements Manger_student, Manger_teacher {
        public Doctor() {

        }

        public Doctor(String name, int age, int number, String sex, double tuition, double salary) {
            this.name = name;
            this.age = age;
            this.number = number;
            this.sex = sex;
            this.tuition = tuition;
            this.salary = salary;
        }

        private String name;
        private int age;
        private int number;
        private String sex;
        private double tuition;
        private double salary;


        public void setName(String name) {
            this.name = name;
        }

        public void setAge(int age) {
            this.age = age;
        }

        public void setNumber(int number) {
            this.number = number;
        }

        public void setSex(String sex) {
            this.sex = sex;
        }

        public void setTuition(double tuition) {
            this.tuition = tuition;
        }

        public void setSalary(double salary) {
            this.salary = salary;
        }

        public String getName() {
            return name;
        }

        public int getAge() {
            return age;
        }

        public String getSex() {
            return sex;
        }

        public double getTuition() {
            return tuition;
        }

        public int getNumber() {
            return number;
        }

        public double getSalary() {
            return salary;
        }


        public double find_tuition() {
            System.out.println("每年学费：" + tuition);
            return tuition;
        }

        public double find_salary() {
            System.out.println("每月工资：" + salary);
            return salary;
        }

        public double afford_tuition() {
            System.out.println("缴纳成功，已缴纳学费" + tuition);
            return tuition;
        }

        public double get_salary() {
            double c;
            c = salary - (salary - 800) * STANDARD;
            System.out.println("薪水已经发放，发放金额：" + c);
            return salary;
        }

        public void taxation() {
            double a;
            a = 12 * ((salary - 800) * STANDARD);
            System.out.println("每年应纳税为：" + a);
        }
    }
}
