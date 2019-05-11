# Datawhale 系列数据结构
## Task1.1  数组

#### 1.1.1实现一个支持动态扩容的数组

```java
	public class EnsureCapacityArray {

	private static final Object[] EMPTY_ELEMENTDATA = {};

	private static final Object[] DEFAULTCAPACITY_EMPTY_ELEMENTDATA = {};
	
	transient Object[] elementData; 
	//传入固定值的情况
	public EnsureCapacityArray(int initialCapacity) {
        if (initialCapacity > 0) {
            this.elementData = new Object[initialCapacity];
        } else if (initialCapacity == 0) {
            this.elementData = EMPTY_ELEMENTDATA;
        } else {
            throw new IllegalArgumentException("Illegal Capacity: "+
                                               initialCapacity);
        }
    }
	//没有传入数组大小的情况
	public EnsureCapacityArray() {
        this.elementData = DEFAULTCAPACITY_EMPTY_ELEMENTDATA;
    }
	
	private static final int MAX_ARRAY_SIZE = Integer.MAX_VALUE - 8;
	/**
	 * 扩容
	 * @param minCapacity
	 */
	public void grow(int minCapacity) {
        int oldCapacity = elementData.length;
        int newCapacity = oldCapacity + (oldCapacity >> 1);
        if (newCapacity - minCapacity < 0)
            newCapacity = minCapacity;
        if (newCapacity - MAX_ARRAY_SIZE > 0)
            newCapacity = hugeCapacity(minCapacity);
        // minCapacity is usually close to size, so this is a win:
        elementData = Arrays.copyOf(elementData, newCapacity);
    }
	//数组容量最大的情况
	private static int hugeCapacity(int minCapacity) {
        if (minCapacity < 0) // overflow
            throw new OutOfMemoryError();
        return (minCapacity > MAX_ARRAY_SIZE) ?
            Integer.MAX_VALUE :
            MAX_ARRAY_SIZE;
    }
}
```
#### 1.1.2 实现一个大小固定的有序数组，支持动态增删改操作
这里理解，大小固定的有序数组，那也就是说数组大小和内部的元素个数完全相同。那么增的时候，需要扩容，删的时候，需要减容。然后数组有序，也就是说，改的时候需要重新排序。
```java
public class FixedArray{
	
	private static final int DEFAULT_SIZE = 10;
	private static final Object[] EMPTY_ELEMENTDATA = {};
	private static final Object[] DEFAULT_ELEMENTDATA = new Object[DEFAULT_SIZE];
	
	transient Object[] elementData;
	
	private int size;
	
	public FixedArray(){
		this.elementData=DEFAULT_ELEMENTDATA;
	}
	
	public FixedArray(int initialCapacity){
		if (initialCapacity > 0) {
            this.elementData = new Object[initialCapacity];
        } else if (initialCapacity == 0) {
            this.elementData = EMPTY_ELEMENTDATA;
        } else {
            throw new IllegalArgumentException("Illegal Capacity: "+
                                               initialCapacity);
        }
	}
	//保证增和删时数组长度和元素长度一致
	private void ensureCapacityInternal(int minCapacity ){
		if (minCapacity - elementData.length > 0)
            grow(minCapacity);
	}
	
	private static final int MAX_ARRAY_SIZE = Integer.MAX_VALUE - 8;
	public void grow(int minCapacity) {
        int oldCapacity = elementData.length;
        int newCapacity = oldCapacity + (oldCapacity >> 1);
        if (newCapacity - minCapacity < 0)
            newCapacity = minCapacity;
        if (newCapacity - MAX_ARRAY_SIZE > 0)
            newCapacity = hugeCapacity(minCapacity);
        // minCapacity is usually close to size, so this is a win:
        elementData = Arrays.copyOf(elementData, newCapacity);
    }
	//数组容量最大的情况
	private static int hugeCapacity(int minCapacity) {
        if (minCapacity < 0) // overflow
            throw new OutOfMemoryError();
        return (minCapacity > MAX_ARRAY_SIZE) ?
            Integer.MAX_VALUE :
            MAX_ARRAY_SIZE;
    }
	//去掉删除之后，重新排序的，空的数组
	public void trimToSize() {
        if (size < elementData.length) {
            elementData = (size == 0)
              ? EMPTY_ELEMENTDATA
              : Arrays.copyOf(elementData, size);
        }
    }
	//判断数组长度
	public int size(){
		return size;
	}
	//判断数组是否为空
	public boolean isEmpty() {
        return size == 0;
    }
	
	//排序
	@SuppressWarnings("unchecked")
	private <E> void sort() {
        Arrays.sort((E[]) elementData, 0, size);
    }
	
	//添加
	public <E> boolean add(E e) {
        ensureCapacityInternal(size + 1);  
        elementData[size++] = e;
        sort();
        return true;
    }
	//删除
	public <E> boolean remove(int index) {
		if (index >= size)
	            throw new IndexOutOfBoundsException();
		
		System.arraycopy(elementData, index+1, elementData, index,
                size-1);
        size--;
        trimToSize();
        return true;
    }
	//改
	public <E> boolean set(int index,Object e) {  
        elementData[index] = e;
        sort();
        return true;
    }
	//查
	public Object get(int index) {     
        return elementData[index];
    }
}
```
#### 1.1.3 合并两个有序的数组
```java
public static int[] mergeArrays(int[]a ,int[]b){
		int alen=a.length;
		int blen=b.length;
		int aindex=0;
		int bindex=0;
		int [] re=new int[alen+blen];
		for(int i=0;i<alen+blen;i++){
			if(aindex<alen && bindex<blen){
				re[i]=a[aindex] > b[bindex] ? b[bindex++] :a[aindex++];
				continue;
			}
			if(aindex==alen && bindex<blen )
				re[i]=b[bindex++];
			if(bindex==blen &&aindex<alen)
				re[i]=a[aindex++];
		}
		return re;
	}
```
#### 1.1.4学习哈希表思想，并完成leetcode上的两数之和(1)及Happy Number(202)！
**哈希表**

```java
//两数之和
public int[] twoSum(int[] nums, int target) {
    Map<Integer, Integer> map = new HashMap<>();
    for (int i = 0; i < nums.length; i++) {
        int complement = target - nums[i];
        if (map.containsKey(complement)) {
            return new int[] { map.get(complement), i };
        }
        map.put(nums[i], i);
    }
    throw new IllegalArgumentException("No two sum solution");
}

//Happy Number(202)
class Solution {
    public boolean isHappy(int n) {
        if(n==1) return true;
        while(n!=1 && n!=4){
            int a=0;
            int ans=0;
            while(n>0){
                a=n%10;
                ans+=a*a;
                n /=10;
            }
            n=ans;
        }
        if(n==1) return true;
        else return false;
    }
}
```

#### 1.1.5 练习
```java
//Three Sum
public List<List<Integer>> threeSum(int[] num) {
    Arrays.sort(num);
    List<List<Integer>> res = new LinkedList<>(); 
    for (int i = 0; i < num.length-2; i++) {
        if (i == 0 || (i > 0 && num[i] != num[i-1])) {
            int lo = i+1, hi = num.length-1, sum = 0 - num[i];
            while (lo < hi) {
                if (num[lo] + num[hi] == sum) {
                    res.add(Arrays.asList(num[i], num[lo], num[hi]));
                    while (lo < hi && num[lo] == num[lo+1]) lo++;
                    while (lo < hi && num[hi] == num[hi-1]) hi--;
                    lo++; hi--;
                } else if (num[lo] + num[hi] < sum) lo++;
                else hi--;
           }
        }
    }
    return res;
}

//Majority Element
public int majorityElement(int[] nums) {
        int res=0;
        int cnt=0;
        for(int num:nums){
            if(cnt==0){
                res=num;
                ++cnt;
            }
            else if(num == res) ++cnt;
            else --cnt;
        }
        return res;
    }

//Missing Positive
 public int firstMissingPositive(int[] nums) {
       if(nums == null&&nums.length==0)
    		return 1;
        int i;
        for(i=0;i<nums.length;i++){
           if(nums[i] != i+1){
               while(0<nums[i]&& nums[i] <= nums.length && nums[nums[i]-1] != nums[i]){
                   int temp = nums[i];
                	nums[i] = nums[nums[i]-1];
                	nums[temp-1] = temp;
               }
           }
       }
        for(i=0;i<nums.length;i++) {
        	if(nums[i]!=i+1) {
        		return i+1;
        	}
        }
        return i+1;
    }
```

## TASK1.2 链表
#### 1.2.1 实现单链表、循环链表、双向链表，支持增删操作
```java
//单链表
class singleNode{
	public int data;
	public singleNode next;
	
	public singleNode head =null;
	
	public singleNode(int data){
		this.data=data;
	}
	
	public void addNode(singleNode node){
		singleNode newNode = new singleNode(data);
        if(head == null){
            head = newNode;
            return;
        }
        singleNode temp = head;
        while(temp.next != null){
            temp = temp.next;
        }
        temp.next = newNode;
	}
	
	public boolean removeNode(int index){
		if(index<1 || index>length()){
			return false;
		}
		if(index == 1){//删除头结点
            head = head.next;
            return true;
        }
        singleNode preNode = head;
        singleNode curNode = preNode.next;
        int i = 1;
        while(curNode != null){
            if(i==index){//寻找到待删除结点
                preNode.next = curNode.next;//待删除结点的前结点指向待删除结点的后结点
                return true;
            }
            //当先结点和前结点同时向后移
            preNode = preNode.next;
            curNode = curNode.next;
            i++;
        }
        return true;
	}
	 public int length(){
	        int length = 0;
	        singleNode curNode = head;
	        while(curNode != null){
	            length++;
	            curNode = curNode.next;
	        }
	        return length;
	 }
}
```
```java
//双向链表
//单向循环链表类
public class CycleLinkList implements List {

    Node head; //头指针
    Node current;//当前结点对象
    int size;//结点个数
    
    //初始化一个空链表
    public CycleLinkList()
    {
        //初始化头结点，让头指针指向头结点。并且让当前结点对象等于头结点。
        this.head = current = new Node(null);
        this.size =0;//单向链表，初始长度为零。
        this.head.next = this.head;
    }
    
    //定位函数，实现当前操作对象的前一个结点，也就是让当前结点对象定位到要操作结点的前一个结点。
    //比如我们要在a2这个节点之前进行插入操作，那就先要把当前节点对象定位到a1这个节点，然后修改a1节点的指针域
    public void index(int index) throws Exception
    {
        if(index <-1 || index > size -1)
        {
          throw new Exception("参数错误！");    
        }
        //说明在头结点之后操作。
        if(index==-1)    //因为第一个数据元素结点的下标是0，那么头结点的下标自然就是-1了。
            return;
        current = head.next;
        int j=0;//循环变量
        while(current != head&&j<index)
        {
            current = current.next;
            j++;
        }
        
    }    
    
    @Override
    public void delete(int index) throws Exception {
        // TODO Auto-generated method stub
        //判断链表是否为空
        if(isEmpty())
        {
            throw new Exception("链表为空，无法删除！");
        }
        if(index <0 ||index >size)
        {
            throw new Exception("参数错误！");
        }
        index(index-1);//定位到要操作结点的前一个结点对象。
        current.setNext(current.next.next);
        size--;
    }

    @Override
    public Object get(int index) throws Exception {
        // TODO Auto-generated method stub
        if(index <-1 || index >size-1)
        {
            throw new Exception("参数非法！");
        }
        index(index);
        
        return current.getElement();
    }

    @Override
    public void insert(int index, Object obj) throws Exception {
        // TODO Auto-generated method stub
        if(index <0 ||index >size)
        {
            throw new Exception("参数错误！");
        }
        index(index-1);//定位到要操作结点的前一个结点对象。
        current.setNext(new Node(obj,current.next));
        size++;
    }

    @Override
    public boolean isEmpty() {
        // TODO Auto-generated method stub
        return size==0;
    }
    @Override
    public int size() {
        // TODO Auto-generated method stub
        return this.size;
    }
}
```
```java
//双向循环链表
//单向链表类
public class DoubleCycleLinkList implements List {

    Node head; //头指针
    Node current;//当前结点对象
    int size;//结点个数

    //初始化一个空链表
    public DoubleCycleLinkList() {
        //初始化头结点，让头指针指向头结点。并且让当前结点对象等于头结点。
        this.head = current = new Node(null);
        this.size = 0;//单向链表，初始长度为零。
        this.head.next = head;
        this.head.prior = head;
    }

    //定位函数，实现当前操作对象的前一个结点，也就是让当前结点对象定位到要操作结点的前一个结点。
    public void index(int index) throws Exception {
        if (index < -1 || index > size - 1) {
            throw new Exception("参数错误！");
        }
        //说明在头结点之后操作。
        if (index == -1)
            return;
        current = head.next;
        int j = 0;//循环变量
        while (current != head && j < index) {
            current = current.next;
            j++;
        }

    }

    @Override
    public void delete(int index) throws Exception {
        // TODO Auto-generated method stub
        //判断链表是否为空
        if (isEmpty()) {
            throw new Exception("链表为空，无法删除！");
        }
        if (index < 0 || index > size) {
            throw new Exception("参数错误！");
        }
        index(index - 1);//定位到要操作结点的前一个结点对象。
        current.setNext(current.next.next);
        current.next.setPrior(current);
        size--;
    }

    @Override
    public Object get(int index) throws Exception {
        // TODO Auto-generated method stub
        if (index < -1 || index > size - 1) {
            throw new Exception("参数非法！");
        }
        index(index);

        return current.getElement();
    }

    @Override
    public void insert(int index, Object obj) throws Exception {
        // TODO Auto-generated method stub
        if (index < 0 || index > size) {
            throw new Exception("参数错误！");
        }
        index(index - 1);//定位到要操作结点的前一个结点对象。
        current.setNext(new Node(obj, current.next));
        current.next.setPrior(current);
        current.next.next.setPrior(current.next);

        size++;
    }

    @Override
    public boolean isEmpty() {
        // TODO Auto-generated method stub
        return size == 0;
    }

    @Override
    public int size() {
        // TODO Auto-generated method stub
        return this.size;
    }
}
```

#### 1.2.2 实现单链表反转
```java
public ListNode reverseList(ListNode head) {
    if (head == null || head.next == null) return head;
    ListNode p = reverseList(head.next);
    head.next.next = head;
    head.next = null;
    return p;
}
```
#### 1.2.3 实现两个有序的链表合并为一个有序链表
```java
 ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if(l1 == NULL) return l2;  
        if(l2 == NULL) return l1;  
        if(l1->val < l2->val) {  
            l1->next = mergeTwoLists(l1->next, l2);  
            return l1;  
        } else {  
            l2->next = mergeTwoLists(l2->next, l1);  
            return l2;  
        }  
    }
```
#### 1.2.4 实现求链表的中间结点
```java
public ListNode middleNode(ListNode head) {
        ListNode fast=head;
        ListNode low=head;
        while(fast != null && fast.next != null){
            low = low.next;
            fast= fast.next.next;
        }
        return low;
    }
```
#### 1.2.5 练习
```java
//Linked List Cycle I（环形链表）
public boolean hasCycle(ListNode head) {
        ListNode f=head;
        ListNode s=head;
        if(head == null || head.next == null) return false;
        while(f!=null && f.next!=null){
            f=f.next.next;
            s=s.next;
            if(f == s) return true;
        }
        return false;
    }
//Merge k Sorted Lists（合并 k 个排序链表）
 public ListNode mergeKLists(ListNode[] lists){
        if(lists.length == 0)
            return null;
        if(lists.length == 1)
            return lists[0];
        if(lists.length == 2){
           return mergeTwoLists(lists[0],lists[1]);
        }

        int mid = lists.length/2;
        ListNode[] l1 = new ListNode[mid];
        for(int i = 0; i < mid; i++){
            l1[i] = lists[i];
        }

        ListNode[] l2 = new ListNode[lists.length-mid];
        for(int i = mid,j=0; i < lists.length; i++,j++){
            l2[j] = lists[i];
        }

        return mergeTwoLists(mergeKLists(l1),mergeKLists(l2));

    }
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) return l2;
        if (l2 == null) return l1;

        ListNode head = null;
        if (l1.val <= l2.val){
            head = l1;
            head.next = mergeTwoLists(l1.next, l2);
        } else {
            head = l2;
            head.next = mergeTwoLists(l1, l2.next);
        }
        return head;
    }

```