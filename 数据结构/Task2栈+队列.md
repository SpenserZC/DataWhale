# Datawhale 系列数据结构
## Task2.1  栈

#### 2.1.1用数组实现一个顺序栈
```java
public class ArrayStack<T> {
	private T [] data;
	private int size;
	private int cnt;
	
	@SuppressWarnings("unchecked")
	public ArrayStack(){
		data= (T[]) new Object [10]; 
		cnt = 0;
		size =10;
	}
	
	public void push(T t){
		if(cnt>=size){
			data=Arrays.copyOf(data, data.length*2);
			size*=2;
		}
		data[size] = t;
		cnt++;
	}
	public T peek(){
		if (cnt==0) {
			throw new EmptyStackException();
		}
		return data[cnt];
	}
	public T pop(){
		if (cnt==0) {
			throw new EmptyStackException();
		}
		cnt--;
		return data[cnt];
	}
	public int search(T t){
		for(int i=cnt;i>0;i--){
			if(data[i]==t)
				return i;
		}
		return -1;
	}
	public boolean isEmpty(){
		return cnt==0;
	}
}
```
#### 2.1.2 用链表实现一个链式栈
```java
public class ListStack<T> {
	private List<T> data;
	private int cnt;
	
	public ListStack(){
		data= new LinkedList<T>(); 
		cnt = 0;
	}
	
	public void push(T t){
		data.add(t);
		cnt++;
	}
	public T peek(){
		if (cnt==0) {
			throw new EmptyStackException();
		}
		return data.get(cnt);
	}
	public T pop(){
		if (cnt==0) {
			throw new EmptyStackException();
		}
		T t=data.remove(cnt);
		cnt--;
		return t; 
	}
	public int search(T t){
		for(int i=cnt;i>0;i--){
			if(data.get(i)==t)
				return i;
		}
		return -1;
	}
	public boolean isEmpty(){
		return cnt==0;
	}
}

```
#### 2.1.3 编程模拟实现一个浏览器的前进后退功能
```java
public class BrowserSimula {
	public static void main(String[] args) {
		Browser b1=new Browser();
		b1.open();
		b1.open();
		b1.open();
		b1.open();
		System.out.println(b1.backward());
		System.out.println(b1.backward());
		System.out.println(b1.forward());
		System.out.println(b1.forward());
	}
}

class Browser{
	private Stack<Integer> s1;
	private Stack<Integer> s2;
	int cnt;
	
	Browser(){
		s1=new Stack<Integer>();
		s2=new Stack<Integer>();
		cnt=0;
	}
	/**
	 * 点开一个新的链接
	 */
	public void open(){
		cnt++;
		s1.push(cnt);
	}
	//后退
	public Integer backward(){
		if(cnt==0)
            throw new ArrayIndexOutOfBoundsException(cnt);
		Integer a =s1.pop();
		s2.push(a);
		return s1.peek();
	}
	//前进
	public Integer forward(){
		if(s2.isEmpty())
			throw new ArrayIndexOutOfBoundsException(cnt);
		Integer a =s2.pop();
		s1.push(a);
		return a;
	}
	
}

```
#### 2.1.4 练习
```java
//Valid Parentheses（有效的括号）
 private HashMap<Character, Character> mappings;

  public Solution() {
    this.mappings = new HashMap<Character, Character>();
    this.mappings.put(')', '(');
    this.mappings.put('}', '{');
    this.mappings.put(']', '[');
  }

  public boolean isValid(String s) {
    Stack<Character> stack = new Stack<Character>();

    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      if (this.mappings.containsKey(c)) {
        char topElement = stack.empty() ? '#' : stack.pop();
        if (topElement != this.mappings.get(c)) {
          return false;
        }
      } else {
        stack.push(c);
      }
    }
    return stack.isEmpty();
  }
//Longest Valid Parentheses（最长有效的括号）[作为可选]
/*dp 方法:
我们用 dp[i] 表示以 i 结尾的最长有效括号；

1 当 s[i] 为 ( , dp[i] 必然等于0,因为不可能组成有效的括号;
2 那么 s[i] 为 )
	2.1 当 s[i-1] 为 (，那么 dp[i] = dp[i-2] + 2；
	2.2 当 s[i-1] 为 ) 并且 s[i-dp[i-1] - 1] 为 (，那么 dp[i] = dp[i-1] + 2 + dp[i-dp[i-1]-2]；
时间复杂度：O(n)
*/
public int longestValidParentheses(String s) {
    if (s == null || s.length() == 0) return 0;
    int[] dp = new int[s.length()];
    int res = 0;
    for (int i = 0; i < s.length(); i++) {
        if (i > 0 && s.charAt(i) == ')') {
            if (s.charAt(i - 1) == '(') {
                dp[i] = (i - 2 >= 0 ? dp[i - 2] + 2 : 2);
            } else if (s.charAt(i - 1) == ')' && i - dp[i - 1] - 1 >= 0 && s.charAt(i - dp[i - 1] - 1) == '(') {
                dp[i] = dp[i - 1] + 2 + (i - dp[i - 1] - 2 >= 0 ? dp[i - dp[i - 1] - 2] : 0);
             }
    	}
    res = Math.max(res, dp[i]);
    }
    return res;
}

//Evaluate Reverse Polish Notatio（逆波兰表达式求值）
 public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        for (int i = 0 ;i < tokens.length;i++){
            String str = tokens[i];
            if (str.length() == 1){
                char ch = str.charAt(0);
                if (ch-'0' >= 0 && ch - '0' <= 9 ){
                    Integer a = Integer.valueOf(str);
                    stack.push(a);
                }                    
                else{
                    if (stack.size() < 2)
                        return 0;
                    int num2 = stack.pop();
                    int num1 = stack.pop();
                    switch(ch){
                        case '+':                            
                            stack.push(num1 + num2);
                            break;
                        case '-':
                            stack.push(num1 - num2);
                            break;
                        case '*':
                            stack.push(num1 * num2);
                            break;
                        case '/':
                            stack.push(num1 / num2);
                            break;
                    }
                    
                }
            }else{
                int n = Integer.valueOf(str);
                stack.push(n);
            }
        }
        return stack.pop(); 
    }

```

## TASK2.2 队列
#### 2.2.1 用数组实现一个顺序队列
```java
public class ArrayQueue<T>  {
	private T [] data ;
	private int cnt;//元素个数	
	private int size;//队列长度
	
	@SuppressWarnings("unchecked")
	public ArrayQueue(){
		data =(T[]) new Object [10];
		cnt = 0;
		size = 10;
	}
	@SuppressWarnings("unchecked")
	public ArrayQueue(int size){
		data =(T[]) new Object [size];
		cnt = 0;
		this.size = size;
	}
	
	public void add(T t){
		if(cnt>=size){
			throw new IllegalStateException();
		}
		data[cnt]=t;
		cnt++;
	}
	
	public T remove(){
		if(cnt<0){
			throw new NoSuchElementException();
		}
		T t= data[0];
		data = Arrays.copyOfRange(data,0,size);
		cnt--;
		return t;
		
	}	
	public boolean offer(T t){
		if(cnt>=size){
			return false;
		}
		data[cnt]=t;
		cnt++;
		return true;
	}	
	public boolean pull(T t){
		if(cnt<0){
			return false;
		}
		data = Arrays.copyOfRange(data,0,size);
		cnt--;
		return true;
	}
	//返回队列头元素
	public T element(){
		return data[0];	
	}	 
	public boolean isEmpty(){
		return cnt==0 ;
	}
	public boolean isFull(){
		return cnt==size;
	}
}

```

#### 2.2.2 用链表实现一个链式队列
```java
public class ListQueue<T> {
	private List<T> data ;
	private int cnt;//元素个数
	
	private int size;//队列长度(用链表的话这里可以强行定义)
	
	public ListQueue(){
		data =new LinkedList<T>();
		cnt = 0;
		size = 10;
	}

	public void add(T t){
		data.add(t);
		cnt++;
	}
	
	public T remove(){
		if(cnt<0){
			throw new NoSuchElementException();
		}
		T t= data.remove(cnt);
		cnt--;
		return t;
		
	}
	
	public boolean offer(T t){
		if(cnt>=size){
			return false;
		}
		data.add(t);
		cnt++;
		return true;
	}
	
	public boolean pull(T t){
		if(cnt<0){
			return false;
		}
		data.remove(cnt);
		cnt--;
		return true;
	}
	//返回队列头元素
	public T element(){
		return data.get(0);	
	}
	
	public boolean isEmpty(){
		return cnt==0 ;
	}
	public boolean isFull(){
		return cnt==size;
	}
}

```
#### 2.2.3 实现一个循环队列
```java
public class MyCircularQueue {
    private final int capacity;
    private final int[] array;
    private int head = 0;
    private int tail = 0;
    private int count = 0;

    /**
     * Initialize your data structure here. Set the size of the queue to be k.
     */
    public MyCircularQueue(int k) {
        this.capacity = k;
        this.array = new int[this.capacity];
    }

    /**
     * Insert an element into the circular queue. Return true if the operation is successful.
     */
    public boolean enQueue(int value) {
        //队列已满
        if (count == capacity) {
            return false;
        }

        //队列为空, 重新设置头部
        if (count == 0) {
            head = (head == capacity) ? 0 : head;
            tail = head;
            array[head] = value;
            count++;
            return true;
        }

        //队列未满 (有空位)
        if (tail == capacity - 1) {
            //tail 达到 maxIndex, 重置为 0
            tail = 0;
        } else {
            //tail 未达到 maxIndex, tail++
            tail++;
        }
        array[tail] = value;
        count++;
        return true;
    }

    /**
     * Delete an element from the circular queue. Return true if the operation is successful.
     */
    public boolean deQueue() {
        if (count == 0) {
            //队列为空
            return false;
        }
        count--;
        head++;
        return true;
    }

    /**
     * Get the front item from the queue.
     */
    public int Front() {
        if (count == 0 ) {
            return -1;
        }
        return array[head];
    }

    /**
     * Get the last item from the queue.
     */
    public int Rear() {
        if (count == 0 ) {
            return -1;
        }
        return array[tail];
    }

    /**
     * Checks whether the circular queue is empty or not.
     */
    public boolean isEmpty() {
        return count == 0;
    }

    /**
     * Checks whether the circular queue is full or not.
     */
    public boolean isFull() {
        return count == capacity;
    }
}

```

#### 2.2.4 练习
```java
//Design Circular Deque（设计一个双端队列）
class MyCircularDeque {
    public int k;
    public int[] numbers;
    public int head;
    public int tail;
    /** Initialize your data structure here. Set the size of the deque to be k. */
    public MyCircularDeque(int k) {
        numbers=new int[k+1];
        head=0;
        tail=0;
        this.k=k;
    }
    
    /** Adds an item at the front of Deque. Return true if the operation is successful. */
    public boolean insertFront(int value) {
        if(isFull())
            return false;
        else{
            head=(head+k)%(k+1);
            numbers[head]=value;
            return true;
        }
    }
    
    /** Adds an item at the rear of Deque. Return true if the operation is successful. */
    public boolean insertLast(int value) {
        if(isFull())
            return false;
        else{
            numbers[tail]=value;
            tail=(tail+1)%(k+1);
            return true;
        }
    }
    
    /** Deletes an item from the front of Deque. Return true if the operation is successful. */
    public boolean deleteFront() {
         if(isEmpty())
            return false;
        else{
            head=(head+1)%(k+1);
            return true;
        }
    }
    
    /** Deletes an item from the rear of Deque. Return true if the operation is successful. */
    public boolean deleteLast() {
         if(isEmpty())
            return false;
        else{
            tail=(tail+k)%(k+1);
            return true;
        }
    }
    
    /** Get the front item from the deque. */
    public int getFront() {
        if(isEmpty())
            return -1;
        return numbers[head];
    }
    
    /** Get the last item from the deque. */
    public int getRear() {
        if(isEmpty())
            return -1;
        return numbers[(tail+k)%(k+1)];
    }
    
    /** Checks whether the circular deque is empty or not. */
    public boolean isEmpty() {
        return tail==head;
    }
    
    /** Checks whether the circular deque is full or not. */
    public boolean isFull() {
        return (tail+1)%(k+1)==head;
    }
}
//Sliding Window Maximum（滑动窗口最大值）
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        if(k==0){
            return new int[0];
        }
        List<Integer> list = new ArrayList<>();//存放窗口内的数字
        int max = Integer.MIN_VALUE;//窗口内的最大数字
        for(int i = 0; i<k;i++){
            if(max<nums[i]){
                max = nums[i];
            }
            list.add(nums[i]);
        }
        int[] res = new int[nums.length - k + 1];//要返回的结果数据
        res[0] = max;
        for(int i = k; i < nums.length;i++){
            int z =list.remove(0);//移走第一位数并插入新的一位数
            list.add(nums[i]);
            if(z!=max){//移走的数不是max，只需判断max与新插入的数哪个大
                if(nums[i]> max){
                    max = nums[i];
                }
                res[i-k+1] = max;
            }else{//移走的数是max，重新判断列表中哪个数是最大的
                if(!list.contains(max)){
                    max = Integer.MIN_VALUE;
                    for(Integer num : list){
                        if(max<num){
                            max = num;
                        }
                    }
                }else{
                    if(nums[i]> max){
                        max = nums[i];
                    }
                }    
            }
            res[i-k+1] = max;
        }
        return res;
    }
}
```

## 3 递归
#### 3.1 编程实现斐波那契数列求值 f(n)=f(n-1)+f(n-2)
```java
public static int Fibe(int n){
		if(n==1) return 1;
		if(n==2) return 1;
		return Fibe(n-1)+Fibe(n-2);
	}
```
#### 3.2 编程实现求阶乘 n!
```java
public static int factorial(int n){
		if(n==1) return 1;
		return factorial(n-1)*n;
	}
```
#### 3.3 编程实现一组数据集合的全排列
```java
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    
    public List<List<Integer>> permute(int[] nums) {
        permition(nums, 0, nums.length-1);
        return res;
    }
    
    public void permition(int[] nums, int p, int q){
        if(p==q){
            res.add(arrayToList(nums));
        }
        for(int i = p; i <= q; i++){
            swap(nums, i, p);
            permition(nums, p+1, q);
            swap(nums, i,p);
        }
    }
    
    private List<Integer> arrayToList(int[] nums){
        List<Integer> res = new ArrayList<>();
        for(int i = 0; i < nums.length; i++){
            res.add(nums[i]);
        }
        
        return res;
    }
    
    private void swap(int[] nums, int i, int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```
#### 3.4 练习
```java
// Climbing Stairs（爬楼梯）
//直接递归，但是直接递归leetcode竟然不能通过，于是又方法2
class Solution {
    public int climbStairs(int n) {
        if(n==1) return 1;
        if(n==2) return 2;
        return  climbStairs(n-1)+climbStairs(n-2);
    }
}

//非递归放啊
class Solution {
    public int climbStairs(int n) {
        int [] ways = new int[n+1];
        ways[0] = 0;
        for (int i = 1;i<ways.length;i++){
            if (i < 3 ){
                ways[i] = i;
            }else {
                ways[i] = ways[i-1] + ways[i-2];
            }
        }
        return ways[n];
    }
}

```