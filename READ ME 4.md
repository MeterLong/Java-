java  
==========================
第四次实验  
------------------------
学生作业处理  
## 实验目的  
1、 掌握字符串String及其方法的使用  
2、 掌握文件的读取/写入方法  
3、 掌握异常处理结构  
## 实验内容  
### 在某课上,学生要提交实验结果，该结果存储在一个文本文件A中。  
    文件A包括两部分内容： 
    1、 是学生的基本信息； 
    2、是学生处理后的作业信息，该作业的业务逻辑内容是：利用已学的字符串处理知识编程完成《长恨歌》古诗的整理对齐工作，写出功能方法，实现如下功能：  
    （1）、 每7个汉字加入一个标点符号，奇数时加“，”，偶数时加“。”  
    （2）、允许提供输入参数，统计古诗中某个字或词出现的次数  
    （3）、输入的文本来源于文本文件B读取，把处理好的结果写入到文本文件A  
    （4）、考虑操作中可能出现的异常，在程序中设计异常处理程序  

## 实验要求  
1、 设计学生类（可利用之前的）；  
2、 采用交互式方式实例化某学生；  
3、 设计程序完成上述的业务逻辑处理，并且把“古诗处理后的输出”结果存储到学生基本信息所在的文本文件A中。

## 实验设计  
    设计思路：
    1、 学生类设计，（用的是实验二的学生类，成员变量有略微的调整）  
    2、 利用字符串的一些方法实现对古诗的格式化，以及添加字符奇数句子加“，”偶数句子加“。\n”实现换行以及分割。  
    3、 设计处理文件类，里面包含：   
     （1）、 读取文件的方法，将文件读取并存入一个字符串  
     （2）、 利用2中的方法处理字符串  
     （3）、 将处理好的字符串写入文件A   
    4、 设计字符输入和统计，利用遍历函数和equals函数进行统计个数，利用input函数进行键盘输入  
    5、 设计主函数，调用上面的函数，以及编辑显示信息，使学生信息、作业处理情况以及字符统计结果输出  

## 主要函数代码  
### 导包  
```
import java.io.*;
import java.util.Scanner;
```
### 学生类  
```
class Student{
    public Student(){

    }
    public Student(String name,int age,int number,String sex){
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

```
### 字符串处理类
```
class Homework{
    public static StringBuffer HW(StringBuffer str1){
        StringBuffer str2 = new StringBuffer(str1);
        int j=0;int z;
        for(int i = 7;i < str2.length()-45;i= i+7){
            z = i +j;
            if(i%2 == 0){
                str2.insert(z,"。\n");
                j = j+2;
            }
            else{
                str2.insert(z,"，");
                j= j+1;
            }
        }
        System.out.println(str2);
        return str2;
    }

}

```

### 文件处理类
```
class Handle{
    public static StringBuffer Read(){
        String str1 = "";
        StringBuffer str2 = new StringBuffer(str1);
        try {
            File file = new File("C:\\Java\\File\\B.txt");
            FileReader fr = new FileReader(file);
            BufferedReader bReader = new BufferedReader(fr);
            while ((str1 =bReader.readLine()) != null) {
                str2.append(str1);
            }
            fr.close();


        } catch (IOException e) {
            e.printStackTrace();
        }
        return  str2;
    }

```
### 字符统计类
```
 public static void Write(StringBuffer s){
        try{
            BufferedWriter bw = new BufferedWriter(new FileWriter("C:\\Java\\File\\A.txt"));
            bw.write(s.toString());
            bw.close();
        }catch (IOException e){
        }
    }

    public static int getCharMaps(String z,StringBuffer s) {
        boolean p;
        int w = 0;
        String a = s.toString();
        char[] b = a.toCharArray();
        for (int i = 0; i < a.length(); i++) {
            String x = String.valueOf(b[i]);
            p = z.equals(x);
            if(p == true){
                w = w + 1;
            }

        }
        return w;
    }
}

```
### 主函数
```
public class experiment_5 {
    public static void main(String args[]) {
        Scanner input = new Scanner(System.in);
        int w;
        StringBuffer str2 = null;
        Handle xms = new Handle();
        StringBuffer str1 = xms.Read();
        Homework one = new Homework();
        str2 = one.HW(str1);
        Handle.Write(str2);

        System.out.println("请输入要查询的字符：");
        String z = input.next();
        w = xms.getCharMaps(z,str1);

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

        System.out.println("作业处理完成");
        System.out.println(z + "出现的次数为："+ z + "次");
    }
}

```

### 调试  
1、 首先遇到的问题就是忘记导包了，以至于写了很多函数都飘红老是找不到关键字定义，后来导包后成功解决  
2、 在字符串格式化处理的时候，一开始我的j变量没放进判断里，导致每次只有前两行是7个字，后面的全是6个字加符号，并且偶数句加的字符是两个，所以后来我将j分别放进判断奇数句+1偶数句子+2，成功解决  
3、 StrignBuffer缓冲区多了45个字符，是由于我没设置缓冲区大小导致的，但是由于文本是固定的我就没添加额外的限制缓冲区条件，只不过在循环的时候length-45就可以了  
4、 equals函数里面的参数我试过很多种，char，string，好几种的失败了，最后我采用的是string.equals(str)这种方法返回布尔值变量从而计数字符个数  
5、 文件读取的时候一开始老是读不进来，后来我设置了filebuffer缓冲区成功加载进来并且读取成字符串形式  
6、 键盘输入函数一开始不太会查找了一下百度解决问题  

### 编程感想
这是最后一次Java试验了，但肯定不会是我最后一次编写Java程序，这一学期学习，让我认识到了Java这个语言的严谨性，我被他的魅力深深的吸引住了，我上学期学了C和Python，但是现在如果让我选择一门语言精修下去，我会选择Java，感谢老师一学期的教导，让我在编程里首先有了一个最核心的思想：面向对象的思想，这是一把新世界大门的钥匙，这不同于我以往的编程经历，他更直接，更有目的性，同时队成员变量的隐私属性也让我大开眼界，这直接引出了封装的重要性。我将来要学的还有很多，希望我能在Java编程的道路上越走越远！
