Java 课程
=========
实验二 学生选课模拟系统  
---------
## 实验目的  
初步了解分析系统需求，从学生选课角度了解系统中的实体及其关系，学会定义类中的属性以及方法；  
掌握面向对象的类设计方法（属性、方法）；  
掌握类的继承用法，通过构造方法实例化对象；  
学会使用super()，用于实例化子类；  
掌握使用Object根类的toString（）方法,应用在相关对象的信息输出中。  

## 实验要求  
说明：学校有“人员”，分为“教师”和“学生”，教师教授“课程”，学生选择“课程”。从简化系统考虑，每名教师仅教授一门课程，每门课程的授课教师也仅有一位，每名学生选仅选一门课程。
对象示例：	人员（编号、姓名、性别）  
教师（编号、姓名、性别、所授课程）  
学生（编号、姓名、性别、所选课程）  
课程（编号、课程名称、上课地点、时间）    

## 核心代码

### 学生和老师的主类
···
class People{
    public People(){

    }
    public People(String name,int age,int number,String sex){
        this.name = name;
        this.age = age;
        this.number = number;
        this.sex = sex;
    }

    private String name;
    private int age;
    private int number;
    private String sex;

    public String getName() {
        return name;
    }
    public int getAge() {
        return age;
    }
    public String getSex() {
        return sex;
    }
    public int getNumber() {
        return number;
    }
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

}
···


### 学生子类
```
class Student extends People{
    public Student(){

    }
    public Student(String name,int age,int number,String sex){
        super(name, age, number, sex);
    }
}

```


### 教师子类
```
class Teacher extends People{
    public Teacher(){

    }
    public Teacher(String name,int age,int number,String sex){
        super(name, age, number, sex);
    }
    String lesson_1;
    public String getLesson_1() {
        return lesson_1;
    }
    public void setLesson_1(String lesson_1) {
        this.lesson_1 = lesson_1;
  
  }
  
}

```

### 课程主类
```
class Lesson{
    public Lesson(){

    }
    public Lesson(String name,String time,int number,String place){
        this.name = name;
        this.time = time;
        this.number = number;
        this.place = place;
    }

    private String name;
    private String time;
    private int number;
    private String place;


    public String getName() {
        return name;
    }
    public String getTime() {
        return time;
    }

    public String getPlace() {
        return place;
    }
    public int getNumber() {
        return number;
    }
    public void setName(String name) {
        this.name = name;
    }
    public void setTime(String time) {
        this.time = time;
    }
    public void setNumber(int number) {
        this.number = number;
    }
    public void setPlace(String place) {
        this.place = place;
    }

}

```

### 课程子类
```
class Lesson_1 extends Lesson{
    public Lesson_1(){

    }
    public Lesson_1(String name,String time,int number,String place){
        super(name, time, number, place);
    }
    public String toString() {
        return "课程名称：" + getName()+ "\n" + "上课时间：" + getTime() + "\n" + "课程编号：" + getNumber()+ "\n" + "授课地点：" + getPlace()+ "\n";
    }
}

```

### 主函数
```

public class Xueshengxuanke {
    public static void main(String[] args) {
        System.out.println("******************学生信息*********************");
        Student xm = new Student();
        xm.setName("王小明");
        xm.setAge(20);
        xm.setNumber(2019666888);
        xm.setSex("男");
        System.out.println("学生姓名:" + xm.getName());
        System.out.println("学生年龄:" + xm.getAge());
        System.out.println("学生编号:" + xm.getNumber());
        System.out.println("学生性别:" + xm.getSex());

        System.out.println("******************教师信息*********************");
        Teacher cdm = new Teacher();
        cdm.setName("陈大妈");
        cdm.setAge(55);
        cdm.setNumber(2001300001);
        cdm.setSex("女");
        cdm.setLesson_1("线性代数");
        System.out.println("授课教师:" + cdm.getName());
        System.out.println("教师年龄:" + cdm.getAge());
        System.out.println("教师编号:" + cdm.getNumber());
        System.out.println("教师性别:" + cdm.getSex());

        System.out.println("******************课程信息*********************");
        Lesson_1 xxds = new Lesson_1();
        xxds.setName("线性代数");
        xxds.setTime("每周三 上午9:40-11:15");
        xxds.setNumber(806050);
        xxds.setPlace("教100");
        System.out.println(xxds.toString());

        System.out.println("******************课程详情*********************");
        System.out.println("恭喜您选课成功，即将开启地狱学期");

    }

}

```

## 编程思路  
1、 首先创建一个人员的父类，为教师和学生子类提供属性和函数参考，拥有基本的人员属性  
2、 创建学生子类，利用super函数进行继承人员父类的属性和方法，从而更好更细致的分类为学生  
3、 创建学生子类，利用super函数进行继承人员父类的属性和方法，从而更好更细致的分类为教师  
4、 创建课程父类，为所有课程子类提供所有课程共有的属性和函数方法  
5、 创建课程子类，利用super函数进行继承人员父类的属性和方法，从而更好更细致的分类为某一课程，并且运用toString()函数，使得在主函数中可以更简单的打印所需信息  
6、 创建主函数，在主函数中对学生、教师、课程进行实例化，并打印显示

## 调试与问题
1、 先便出现了toString()，在静态函数中不兼容问题，后来重新定义了toString()函数，得以解决  
2、 子类没有很好地继承父类属性以及方法，后来运用了super()函数，问题得以解决  
3、 在主函数调试过程中有一两个属性一直无法实例化，得到默认属性null，后来经检查，get函数中返回值类型出错，改正后问题解决了  
4、 在toString()函数中一开始无法换行，后来熟悉语法加入"\n"，成功换行

## 实验感想
在本次实验过程中，初步了解到了学生选课系统的结构和框架，虽然我们只是做出了最简化版本，但依旧让我受益匪浅，我在本次实验中对this、toString、super这三个函数有了进一步的理解，同时在编译过程中也找到了一些方法和技巧，让我不再感觉特别吃力，另外我在这次编程中也联想到了C语言和Python的一些相同的语言特性，以及对编译器有了初步的掌握，这些让我有信心在Java编程的道路上越走越远。
