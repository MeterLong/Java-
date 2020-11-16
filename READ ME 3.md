Java  
=============  
Java课程作业  实验三  
------------------  
接口与异常处理  

###  实验目的
1、 掌握Java中抽象类和抽象方法的定义；   
2、 掌握Java中接口的定义，熟练掌握接口的定义形式以及接口的实现方法  
3、 了解异常的使用方法，并在程序中根据输入情况做异常处理  

###  实验内容
某学校为了给学生提供勤工俭学机会，也减轻授课教师的部分压力，准许博士研究生参与课程的助教工作。此时，该博士研究生有双重身份：学生和助教教师。  
1、 设计两个管理接口：学生管理接口和教师管理接口。学生接口必须包括缴纳学费、查学费的方法；教师接口包括发放薪水和查询薪水的方法。  
2、 设计博士研究生类，实现上述的两个接口，该博士研究生应具有姓名、性别、年龄、每学期学费、每月薪水等属性。（其他属性及方法，可自行发挥）  
3、 编写测试类，并实例化至少两名博士研究生，统计他们的年收入和学费。根据两者之差，算出每名博士研究生的年应纳税金额（国家最新工资纳税标准，请自行检索）。  
######  要求：
1、 在博士研究生类中实现各个接口定义的抽象方法;  
2、 对年学费和年收入进行统计，用收入减去学费，求得纳税额；  
3、 国家最新纳税标准（系数），属于某一时期的特定固定值，与实例化对象没有关系，考虑如何用static  final修饰定义。  
4、 实例化研究生类时，可采用运行时通过main方法的参数args一次性赋值，也可采用Scanner类实现运行时交互式输入。  
5、 根据输入情况，要在程序中做异常处理。  

###  实验设计
1、 创建学生管理接口，里边创建两个抽象方法，分别为查询学费和缴纳学费  
2、 创建教师管理接口，先建立纳税率常量，再里边创建两个抽象方法，分别为工资查询和工资发放  
3、 创建研究生类继承上面两个接口，在研究生类里面定义几个变量（姓名，性别，学号，年龄，学费，工资等），并建立set和get方法用于设定数值和获取数值，
并将上面两个接口的方法全部重写来进行接口抽象方法的实现，设定纳税方法，并将结果打印  
4、 创建主测试函数，首先利用try，catch的异常处理函数进行异常预处理，并将两名研究生的信息实例化，并且调用纳税方法，最后打印所有学生信息的结果  

###  运行结果
```
C:\Java\bin\java.exe "-javaagent:C:\Java\Java_IJ\IntelliJ IDEA 2020.2.3\lib\idea_rt.jar=58537:C:\Java\Java_IJ\IntelliJ IDEA 2020.2.3\bin" -Dfile.encoding=UTF-8 -classpath C:\Java\Test__1\out\production\Test__0 Test_1
******************研究生一*********************
学生姓名:王小明
学生年龄:20
学生编号:2019666888
学生性别:男
每年学费：8000.0
每月工资：1200.0
每年应纳税为：960.0
缴纳成功，已缴纳学费8000.0
薪水已经发放，发放金额：1120.0
******************研究生二*********************
学生姓名:董晓芳
学生年龄:20
学生编号:2019666999
学生性别:女
每年学费：8000.0
每月工资：1185.0
每年应纳税为：924.0
缴纳成功，已缴纳学费8000.0
薪水已经发放，发放金额：1108.0

Process finished with exit code 0

```

###  核心代码  
#  学生管理接口
```
interface Manger_student {
        double find_tuition();

        double afford_tuition();
    }
```

#  教师管理接口
```
interface Manger_teacher {
        double STANDARD = 0.2;

        double find_salary();

        double get_salary();
    }

```
#  研究生类
```
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
```

#  主测试函数
```
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
```

###  调试
1、 在创建研究生类继承两个接口时出现了语法错误，没有同时继承两个接口，修改后可继承  
2、 在创建研究生类继承两个接口时没有将两个接口全部方法重写只实现了一部分，报错，在实现全部方法后成功通过编译  
3、 try函数一开始 包住了主函数编译错误，将try，catch函数放入主函数内部成功编译  
4、 写catch函数里面的错误变量总是写不明白，后来改成了Exception，成功编译
5、 设计纳税方法的时候开始税率总是整不明白，经查百度后，又重新赋予变量和打印效果成功运行


###  实验心得
在本次编程的过程中，出现了许多问题并解决，这一点在上边的调试里有详细说明，在过程中我体会了Java和其他语言不同的语法以及设计结构，
尤其是各个接口和类之间的参数调用，接口中的方法全部为抽象方法，如果有子类继承的话需要将这些方法全部重写或者定义来实现这些抽象方法。
在编程的过程中也再次让我回归到了那个严谨的状态，不管是标点符号还是缩进都要严格的按照规范编写，否则就会报错，
在抽象方法和异常处理上我有了一定的了解，让我知道了java在世纪最后弄得几种错误本质区别和一些基础的常见的错误类型，有些我们能改有些我们不能改变。
这次异常处理的学习让我在以后的代码书写中更加完备和硬性，接口的学习让我的代码更加的多态和快捷。

