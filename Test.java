package one;

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

class HardDisk_1 {
	int amount;
	int getAmount(){
		return amount;
	}
	public void setAmount(int amount){
		this.amount = amount;
		}
}

class CPU_1{
	int speed;
	int getSpeed(){
		return speed;
	}
	public void setSpeed(int speed){
		this.speed = speed;
		}
}

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