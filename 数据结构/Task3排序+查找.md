# Datawhale 系列数据结构
## Task3.1  排序

#### 3.1.1归并
```java
//采用分治(Divide and Conquer)的一个非常典型的应用。将已有序的子序列合并，得到完全的序列
	public static int [] mergeSort(int []arr){
		int len =arr.length;
		if(len<2){
			return arr;
		}
		int [] left=Arrays.copyOfRange(arr,0,len/2);
		int [] right=Arrays.copyOfRange(arr,len/2,len);
		return merge(mergeSort(left),mergeSort(right));
	}
	public static int [] merge(int [] left,int [] right){
		int llen=left.length;
		int rlen=right.length;
		int[] res=new int[llen+rlen];
		int li=0,ri=0,rei=0;
		while (llen-li>0 && rlen-ri>0) {
			if (left[li] <= right[ri]) {
	            res[rei++]=left[li++];
	        } else {
	        	res[rei++]=right[ri++];
	        }
	    }
		while (llen-li>0) {
            res[rei++]=left[li++];
	    }
		while (rlen-ri>0) {
            res[rei++]=right[ri++];
	    }
		return res;
	}
```
#### 3.1.2 快速排序
```java
/*快速排序使用分治法来把一个list分为两个子list:
*从数列中跳出一个元素，称为“基准”（pivot）
*重新排序数列，所有元素比基准小的摆放在基准前面，所有比基准大的放在基准后面。在这个分区推出后，该基准就处于数列的中间位置。这个称为分区操作。
*递归的，把小于基准值元素的子数列和大于基准值元素的子数列排序
*/
public static int partition(int []array,int lo,int hi){
        //固定的切分方式
        int key=array[lo];
        while(lo<hi){
            while(array[hi]>=key&&hi>lo){//从后半部分向前扫描
                hi--;
            }
            array[lo]=array[hi];
            while(array[lo]<=key&&hi>lo){从前半部分向后扫描
                lo++;
            }
            array[hi]=array[lo];
        }
        array[hi]=key;
        return hi;
    }
    
    public static void sort(int[] array,int lo ,int hi){
        if(lo>=hi){
            return ;
        }
        int index=partition(array,lo,hi);
        sort(array,lo,index-1);
        sort(array,index+1,hi); 
    }

```
#### 3.1.3 插入
```java
public static int [] insertionSort(int []arr){
		for(int i=0;i<arr.length;i++){
			for(int j=i;j>0;j--){ 
				if(arr[j]<arr[j-1]){
					int temp=arr[j];
					arr[j]=arr[j-1];
					arr[j-1]=temp;
				}
			}
		}
		return arr;
		
	}

```
#### 3.1.4 冒泡
```java
public static int [] bubbleSort(int [] arr){
		for(int i= 0;i<arr.length-1;i++){
			for(int j=0;j<arr.length-1-i;j++){
				if(arr[j]>arr[j+1]){
					int temp =arr[j];
					arr[j]=arr[j+1];
					arr[j+1]=temp;
				}
			}
		}
		return arr;
	}
	
```
#### 3.1.5 选择
```java
public static int [] selectionSort(int [] arr){
		int min = 0;
		for(int i=0;i<arr.length-1;i++){
			for(int j=i+1;j<arr.length-1;j++){
				min = arr[i]; 
				if(arr[j]<min){
					min=arr[j];
					arr[j]=arr[i];
					arr[i]=min;
				}
			}
		}
		return arr;
	}
```
#### 3.1.6 堆排序(选做)
```java
/*堆排序分为三个步骤：
*	创建最大堆
*	确保最大堆中父节点的值比子节点的值都大
*	将根节点与最后一个叶子节点比较，择其大者剔除出堆，再重复第2、3步。
*第二步是整个堆排序的关键。
*/
public static void maxHeapify(int[] array, int heapsize, int i){
    int l = 2*i + 1;
    int r = 2*i + 2;
    int large = i;
    if (l < heapsize && array[i] < array[l]) {
        large = l;
    }else {
        large = i;
    }
    if (r < heapsize && array[large] < array[r]) {
        large = r;
    }
    if (large != i) {
        int temp = array[i];
        array[i] = array[large];
        array[large] = temp;
        //因为将最大值和父节点交换了位置，新的子节点并不能保证一定是比它的子节点大
        //所以需要递归，确定交换的子节点比它的子节点都大
        //而没有动的子节点是不需要进行递归的，因为它的数值没有变，如果之前满足最大堆条件，现在就还是满足的
        maxHeapify(array, heapsize, large);
    }
}
//创建堆
public static void buildMaxHeap(int[] array){
    int heapsize = array.length;
    for (int i = heapsize/2; i >= 0; i--) {
        maxHeapify(array,heapsize,i);
    }
}
public static void heapSort(int[] array){
    int heapsize = array.length;
    for (int i = heapsize - 1; i > 0; i--) {
        if (array[i] < array[0]) {
            int temp = array[0];
            array[0] = array[i];
            array[i] = temp;
            heapsize --;
            maxHeapify(array, heapsize, 0);
        }
    }
}


```
#### 3.1.8 编程实现 O(n) 时间复杂度内找到一组数据第 K 大元素
```java
//采用堆排序的方法
//在创建最小堆，只创建K个元素
public static void maxHeapify(int[] array, int size, int i) {
	    int left = 2 * i + 1;
	    int right = 2 * i + 2;
	    int small = i;
	    if (left < size) {
	        if (array[small] > array[left]) {
	            small = left;
	        }
	    }
	    if (right < size) {
	        if (array[small] > array[right]) {
	            small = right;
	        }
	    }
	    if (small != i) {
	        int temp = array[small];
	        array[small] = array[i];
	        array[i] = temp;
	        maxHeapify(array, size, small);
	    }
	}

	public static void buildHeap(int[] array, int size) {
	    for (int i = size - 1; i >= 0; i--) {
	        maxHeapify(array, size, i);
	    }
	}
	
	public static int findKByHeap(int[] array, int k) {
	    buildHeap(array, k);
	    for (int i = k + 1; i < array.length; i++) {
	        if (array[i] > array[0]) {
	            int temp = array[i];
	            array[i] = array[0];
	            array[0] = temp;
	            maxHeapify(array, k, 0);
	        }
	    }
	    return array[0];
	}

```

## TASK3.2 查找
#### 3.2.1 实现一个有序数组的二分查找
```java
//默认数组是有序数组
	public static int biSearch(int [] arr, int target){
		int r = arr.length-1;
		int l = 0;
		int mid=r/2;
		while(l<=r){
			mid=(l+r)/2;
			if(arr[mid]==target)
				return mid;
			else if(arr[mid]>target)
				r=mid;
			else
				l=mid;
		}
		return -1;
		
		
	} 
```

#### 3.2.2 实现模糊二分查找算法（比如大于等于给定值的第一个元素）
```java
//模糊二分查找，返回大于等于给定值的第一个值的下标
	public static int blurrySearch(int [] arr, int target){
		int r = arr.length-1;
		int l = 0;
		int mid=r/2;
		while(l<=r){
			mid=(l+r)/2;
			if(arr[mid]==target)
				return mid;
			else if(arr[mid]>target)
				r=mid-1;
			else
				l=mid+1;
		}
		return r+1;
	} 
	

```
#### 3.2.3 Sqrt(x)（x的平方根）
```java
class Solution {
    public int mySqrt(int x) {
        if(x==1) return 1;
        int min=0;
        int max = x;
        while(max-min>1){
            int m=(max+min)/2;
            if(x/m<m) max=m;
            else min = m;
        }
        return min;
    
    }
}
```