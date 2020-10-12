# Java-
Java课程作业项目仓库

# 实验内容
用类描述计算机中CPU的速度和硬盘容量。

### 实验方法
1. 利用Eclipse进行源代码的编译、调试及运行
2. 用Github网站进行实验报告的撰写

### 实验设计
1. 首先进行框架设计，在Test主类下有CPU、HardDisk、PC三个分类，主类从分类调取参数并设计和调用显示函数show（）从而达到显示CPU速度以及硬盘容量的目的
2. 创建项目Experiment，并创建包one，在此包下进行类的创建和编写
3. 创建CPU_1类，创建参数speed，并创建方法getSpeed，用于之后为speed赋值
4. 创建HardDisk_1类，创建参数amonut，并创建方法getAmount，用于之后为amount赋值
5. 创建PC类，创建属性cpu和HD，并创建方法getCPU、getHardDisk，为属性赋值，并创建显示函数show（），在函数中利用打印函数对最终的cpu速度和硬盘容量进行显示
6. 最后创建主类Test，首先将Test类里设置主函数，调用上述三类中的函数，并为其赋值、和显示 

#核心代码
1. CPU_1类
```
class CPU_1{
	int speed;
	int getSpeed(){
		return speed;
	}
	public void setSpeed(int speed){
		this.speed = speed;
		}
}
```
2. HardDisk_1类
```
class HardDisk_1 {
	int amount;
	int getAmount(){
		return amount;
	}
	public void setAmount(int amount){
		this.amount = amount;
		}
}
```
3. PC类
```
class PC{
	 CPU_1 cpu;
	 HardDisk_1 HD;
     public void setCPU(CPU_1 cpu) {
    	 this.cpu = cpu;
    	 }
     public void setHardDisk(HardDisk_1 h){
    	 this.HD = h;
     }
     public void show(){
    	 System.out.println("cpu速度为: " +cpu.getSpeed());
         System.out.println("硬盘容量为: "+HD.getAmount());
     }

}
```
4. Test主类
```
public class Test {
	public static void main(String args[]){
		CPU_1 cpu = new CPU_1();
		cpu.setSpeed(2200);
		
		HardDisk_1 disk = new HardDisk_1();
		disk.setAmount(200);
		
		PC pc = new PC();
		pc.setCPU(cpu);
		pc.setHardDisk(disk);
		pc.show();
	}

}
```
# 调试
##出现的问题
1. 在一开始创建设计的时候，本来想一个类别创建一个文件，后来把类都写在一个文件里运行时，出现了重复错误，在同一文件的类名不能和单独类文件的名字相同，于是我将同一文件里面的类名后边加上了_1，解决了此错误
2. 在编写过程中，有的时候忘了加分号又忘记看标示，导致程序运行不成功，后来将分号补上，成功运行
3. 参数不一致，在赋值过程中因为自己的语法错误或者加了个点或者加了个空格造成没有正确调用参数，修改后运行成功
4. 后来尝试在不同的包里进行类别的创建和调用，并对参数属性进行了private加密，使其他类文件不可以调用此参数

# 实验感想
在本次编程的过程中，出现了许多问题并解决，这一点在上边的调试里有详细说明，在过程中我体会了Java和其他语言不同的语法以及设计结构，尤其是各个包和类之间的参数调用。在编程的过程中也再次让我回归到了那个严谨的状态，不管是标点符号还是缩进都要严格的按照规范编写，否则就会报错，在方法和构造方法的区别上我有了一定的了解，在对参数的加密属性上我也有了一定了解。Java真的是一门让我很感兴趣的语言，这便是这次实验带给我的感想



