# Datawhale 系列数据结构
## Task5.1树
#### 5.1.1实现一个二叉查找树（支持插入，删除，查找操作）
```java
public class BSTree<T extends Comparable<T>> {

    private BSTNode<T> mRoot;    // 根结点

    @SuppressWarnings("hiding")
	public class BSTNode<T extends Comparable<T>> {
        T key;                // 关键字(键值)
        BSTNode<T> left;    // 左孩子
        BSTNode<T> right;    // 右孩子
        BSTNode<T> parent;    // 父结点

        public BSTNode(T key, BSTNode<T> parent, BSTNode<T> left, BSTNode<T> right) {
            this.key = key;
            this.parent = parent;
            this.left = left;
            this.right = right;
        }

        public T getKey() {
            return key;
        }

        public String toString() {
            return "key:"+key;
        }
    }

    public BSTree() {
        mRoot=null;
    }

    /*
     * 前序遍历"二叉树"
     */
    private void preOrder(BSTNode<T> tree) {
        if(tree != null) {
            System.out.print(tree.key+" ");
            preOrder(tree.left);
            preOrder(tree.right);
        }
    }

    public void preOrder() {
        preOrder(mRoot);
    }

    /*
     * 中序遍历"二叉树"
     */
    private void inOrder(BSTNode<T> tree) {
        if(tree != null) {
            inOrder(tree.left);
            System.out.print(tree.key+" ");
            inOrder(tree.right);
        }
    }

    public void inOrder() {
        inOrder(mRoot);
    }


    /*
     * 后序遍历"二叉树"
     */
    private void postOrder(BSTNode<T> tree) {
        if(tree != null)
        {
            postOrder(tree.left);
            postOrder(tree.right);
            System.out.print(tree.key+" ");
        }
    }

    public void postOrder() {
        postOrder(mRoot);
    }


    /*
     * (递归实现)查找"二叉树x"中键值为key的节点
     */
    private BSTNode<T> search(BSTNode<T> x, T key) {
        if (x==null)
            return x;

        int cmp = key.compareTo(x.key);
        if (cmp < 0)
            return search(x.left, key);
        else if (cmp > 0)
            return search(x.right, key);
        else
            return x;
    }

    public BSTNode<T> search(T key) {
        return search(mRoot, key);
    }

    /*
     * (非递归实现)查找"二叉树x"中键值为key的节点
     */
    private BSTNode<T> iterativeSearch(BSTNode<T> x, T key) {
        while (x!=null) {
            int cmp = key.compareTo(x.key);

            if (cmp < 0) 
                x = x.left;
            else if (cmp > 0) 
                x = x.right;
            else
                return x;
        }

        return x;
    }

    public BSTNode<T> iterativeSearch(T key) {
        return iterativeSearch(mRoot, key);
    }

    /* 
     * 查找最小结点：返回tree为根结点的二叉树的最小结点。
     */
    private BSTNode<T> minimum(BSTNode<T> tree) {
        if (tree == null)
            return null;

        while(tree.left != null)
            tree = tree.left;
        return tree;
    }

    public T minimum() {
        BSTNode<T> p = minimum(mRoot);
        if (p != null)
            return p.key;

        return null;
    }
     
    /* 
     * 查找最大结点：返回tree为根结点的二叉树的最大结点。
     */
    private BSTNode<T> maximum(BSTNode<T> tree) {
        if (tree == null)
            return null;

        while(tree.right != null)
            tree = tree.right;
        return tree;
    }

    public T maximum() {
        BSTNode<T> p = maximum(mRoot);
        if (p != null)
            return p.key;

        return null;
    }

    /* 
     * 找结点(x)的后继结点。即，查找"二叉树中数据值大于该结点"的"最小结点"。
     */
    public BSTNode<T> successor(BSTNode<T> x) {
        // 如果x存在右孩子，则"x的后继结点"为 "以其右孩子为根的子树的最小结点"。
        if (x.right != null)
            return minimum(x.right);

        // 如果x没有右孩子。则x有以下两种可能：
        // (01) x是"一个左孩子"，则"x的后继结点"为 "它的父结点"。
        // (02) x是"一个右孩子"，则查找"x的最低的父结点，并且该父结点要具有左孩子"，找到的这个"最低的父结点"就是"x的后继结点"。
        BSTNode<T> y = x.parent;
        while ((y!=null) && (x==y.right)) {
            x = y;
            y = y.parent;
        }

        return y;
    }
     
    /* 
     * 找结点(x)的前驱结点。即，查找"二叉树中数据值小于该结点"的"最大结点"。
     */
    public BSTNode<T> predecessor(BSTNode<T> x) {
        // 如果x存在左孩子，则"x的前驱结点"为 "以其左孩子为根的子树的最大结点"。
        if (x.left != null)
            return maximum(x.left);

        // 如果x没有左孩子。则x有以下两种可能：
        // (01) x是"一个右孩子"，则"x的前驱结点"为 "它的父结点"。
        // (01) x是"一个左孩子"，则查找"x的最低的父结点，并且该父结点要具有右孩子"，找到的这个"最低的父结点"就是"x的前驱结点"。
        BSTNode<T> y = x.parent;
        while ((y!=null) && (x==y.left)) {
            x = y;
            y = y.parent;
        }

        return y;
    }

    /* 
     * 将结点插入到二叉树中
     *
     * 参数说明：
     *     tree 二叉树的
     *     z 插入的结点
     */
    private void insert(BSTree<T> bst, BSTNode<T> z) {
        int cmp;
        BSTNode<T> y = null;
        BSTNode<T> x = bst.mRoot;

        // 查找z的插入位置
        while (x != null) {
            y = x;
            cmp = z.key.compareTo(x.key);
            if (cmp < 0)
                x = x.left;
            else
                x = x.right;
        }

        z.parent = y;
        if (y==null)
            bst.mRoot = z;
        else {
            cmp = z.key.compareTo(y.key);
            if (cmp < 0)
                y.left = z;
            else
                y.right = z;
        }
    }

    /* 
     * 新建结点(key)，并将其插入到二叉树中
     *
     * 参数说明：
     *     tree 二叉树的根结点
     *     key 插入结点的键值
     */
    public void insert(T key) {
        BSTNode<T> z=new BSTNode<T>(key,null,null,null);

        // 如果新建结点失败，则返回。
        if (z != null)
            insert(this, z);
    }

    /* 
     * 删除结点(z)，并返回被删除的结点
     *
     * 参数说明：
     *     bst 二叉树
     *     z 删除的结点
     */
    private BSTNode<T> remove(BSTree<T> bst, BSTNode<T> z) {
        BSTNode<T> x=null;
        BSTNode<T> y=null;

        if ((z.left == null) || (z.right == null) )
            y = z;
        else
            y = successor(z);

        if (y.left != null)
            x = y.left;
        else
            x = y.right;

        if (x != null)
            x.parent = y.parent;

        if (y.parent == null)
            bst.mRoot = x;
        else if (y == y.parent.left)
            y.parent.left = x;
        else
            y.parent.right = x;

        if (y != z) 
            z.key = y.key;

        return y;
    }

    /* 
     * 删除结点(z)，并返回被删除的结点
     *
     * 参数说明：
     *     tree 二叉树的根结点
     *     z 删除的结点
     */
    public void remove(T key) {
        BSTNode<T> z, node; 

        if ((z = search(mRoot, key)) != null)
            if ( (node = remove(this, z)) != null)
                node = null;
    }

    /*
     * 销毁二叉树
     */
    private void destroy(BSTNode<T> tree) {
        if (tree==null)
            return ;

        if (tree.left != null)
            destroy(tree.left);
        if (tree.right != null)
            destroy(tree.right);

        tree=null;
    }

    public void clear() {
        destroy(mRoot);
        mRoot = null;
    }

    /*
     * 打印"二叉查找树"
     *
     * key        -- 节点的键值 
     * direction  --  0，表示该节点是根节点;
     *               -1，表示该节点是它的父结点的左孩子;
     *                1，表示该节点是它的父结点的右孩子。
     */
    private void print(BSTNode<T> tree, T key, int direction) {

        if(tree != null) {

            if(direction==0)    // tree是根节点
                System.out.printf("%2d is root\n", tree.key);
            else                // tree是分支节点
                System.out.printf("%2d is %2d's %6s child\n", tree.key, key, direction==1?"right" : "left");

            print(tree.left, tree.key, -1);
            print(tree.right,tree.key,  1);
        }
    }

    public void print() {
        if (mRoot != null)
            print(mRoot, mRoot.key, 0);
    }
}


```
#### 5.1.2 实现查找二叉查找树中某个节点的后继，前驱节点
```java
/**
* 前驱元素
* **/
public BSTreeNode<T> Pred(BSTreeNode<T> node) {
	if (node.left != null) {
		return Max(node.left);
	}
	BSTreeNode<T> parent = node.parent;
	while (parent != null && node != parent.right) {
		node = parent;
		parent = node.parent;
	}
	return parent;
}

/**
* 后继元素
* **/
public BSTreeNode<T> Succ(BSTreeNode<T> node) {
	if (node.right != null) {
		return Min(node.right);
	}
	BSTreeNode<T> parent = node.parent;
	while (parent != null && node != parent.left) {
		node = parent;
		parent = node.parent;
	}
	return parent;
}
```
#### 5.1.3 实现二叉树前，中，后序以及层次遍历
```java
/*
*在5.1.1中已经实现
*/
```
#### 5.1.4 练习：翻转二叉树
```java
class Solution{
	public TreeNode invertTree(TreeNode root){
		if(root==null)
			return root;
		TreeNode temp=root.left;
		root.left=root.right;
		root.right=temp;
		invertTree(root.left);
		invertTree(root.right);
		return root;
	}
}
```

#### 5.1.5 二叉树的最大深度
```java
class Solution {
    public int maxDepth(TreeNode root) {
        if(root==null) return 0;
        return 1+Math.max(maxDepth(root.left),maxDepth(root.right));
        
    }
```
#### 5.1.6 验证二叉查找树
```java
class Solution {
    private TreeNode pre = null;
    public boolean isValidBST(TreeNode root) {
       return helper(root);
    }    
    private boolean helper(TreeNode root){
         if(root == null)
            return true;
         if(!helper(root.left))
             return false;
         if(pre != null && pre.val >= root.val)
             return false;
         pre = root;        
         return helper(root.right);
    }
}

```
## TASK5.2 堆
#### 5.2.1 实现一个小顶堆，大顶堆
```java
public class MaxHeap<T extends Comparable<T>> {
	private List<T> mHeap; // 存放元素的动态数组
 
	public MaxHeap() {
		this.mHeap = new ArrayList<>();
	}
	/**
	 * 大顶堆的向上调整算法(添加节点的时候调用) 注：数组实现的堆中，第N个节点的左孩子的索引值是(2N+1)，右孩子的索引是(2N+2)。
	 * 
	 * @param start
	 *            -- 被上调节点的起始位置(一般为数组中最后一个元素的索引)
	 */
	protected void filterup(int start) {
 
		int c = start; // 需要调整的节点的初始位置
		int p = (c - 1) / 2; // 当前节点的父节点的位置
		T tmp = mHeap.get(c); // 被调整节点的值
 
		while (c > 0) {
			// 父节点的值和被调整节点的值进行比较
			int cmp = mHeap.get(p).compareTo(tmp);
			if (cmp >= 0) {
				// 父节点大
				break;
			} else {
				// 被调整节点的值大，交换
				mHeap.set(c, mHeap.get(p));
				c = p;
				p = (c - 1) / 2;
			}
		}
		// 找到被调整节点的最终位置了
		mHeap.set(c, tmp);
	}
 
	/**
	 * 大顶堆的向下调整算法(删除节点的时候需要调用来调整大顶堆)
	 * 注：数组实现的堆中，第N个节点的左孩子的索引值是(2N+1)，右孩子的索引是(2N+2)。
	 * 
	 * @param start
	 *            -- 被下调节点的起始位置(一般为0，表示从第1个开始)
	 * @param end
	 *            -- 截至范围(一般为数组中最后一个元素的索引)
	 */
	protected void filterdown(int start, int end) {
 
		int c = start; // 被下调节点的初始位置
		int l = 2 * c + 1; // 左孩子节点的位置
		T tmp = mHeap.get(c); // 当前节点的值(大小)
 
		while (l <= end) {
			// 当前节点的左右节点进行比较
			int cmp = mHeap.get(l).compareTo(mHeap.get(l + 1));
			// 取大的
			if (l < end && cmp < 0) {
				l++;
			}
			// 当前节点和大的那个再比较一下
			cmp = tmp.compareTo(mHeap.get(l));
			if (cmp >= 0) {
				// 当前节点大,不用动
				break;
			} else {
				// 当前节点小,交换
				mHeap.set(c, mHeap.get(l));
				c = l; // 更新当前节点的位置
				l = 2 * c + 1; // 更新当前节点的左孩子位置
			}
		}
		mHeap.set(c, tmp);
	}
 
	/**
	 * 向大顶堆中插入新元素
	 * 
	 * @param data
	 */
	public void insert(T data) {
 
		int insertIndex = mHeap.size(); // 获取插入的位置
		// 将新元素插入到数组尾部
		mHeap.add(data);
		// 调用filterup函数，调整大顶堆
		filterup(insertIndex);
	}
	/**
	 * 删除大顶堆中的data节点
	 * 
	 * @param data
	 * @return 返回-1表示出错, 返回0表示删除成功
	 */
	public int remove(T data) {
 
		// 大顶堆空
		if (mHeap.isEmpty()) {
			return -1;
		}
 
		// 获取data在数组中的索引
		int index = mHeap.indexOf(data);
		if (index == -1) {
			return -1;
		}
 
		// 堆中元素的个数
		int size = mHeap.size();
		// 删除了data元素，需要用最后一个元素填补，然后调用filterdown算法进行调整
		mHeap.set(index, mHeap.get(size - 1)); // 用最后一个元素填补
		mHeap.remove(size - 1); // 删除最后一个元素
 
		if (mHeap.size() > 1 && index < mHeap.size()) {
			// 调整成大顶堆
			filterdown(index, mHeap.size() - 1);
		}
		return 0;
	}
	
	
	@Override
	public String toString() {
		
		StringBuilder sb = new StringBuilder();
		for(int i = 0; i < mHeap.size(); i++) {
			sb.append(mHeap.get(i) + " ");
		}
		return sb.toString();
	}
	
	public static void main(String[] args) {
		
		int a[] = {10, 40 ,30, 60, 90, 70, 20, 50 ,80};
		
		//大顶堆
		MaxHeap<Integer> maxHeap = new MaxHeap<>();
		
		//添加元素
		System.out.println("=== 依次添加元素：");
		for(int i = 0; i < a.length; i++) {
			System.out.println(a[i]);
			maxHeap.insert(a[i]);
		}
		
		//生成的大顶堆
		System.out.println("=== 生成的大顶堆：");
		System.out.println(maxHeap);
		
		//添加新元素85
		int data = 85;
		maxHeap.insert(data);
		System.out.println("=== 添加新元素" + data + "之后的大顶堆：");
		System.out.println(maxHeap);
		
		//删除元素90
		data = 90;
		maxHeap.remove(data);
		System.out.println("=== 删除元素" + data + "之后的大顶堆：");
		System.out.println(maxHeap);
	}
}
```
#### 5.2.2 实现优先级队列
```java
/**
 * 优先队列类（最大优先队列）
 */
public class PriorityHeap {

    // ------------------------------ Instance Variables

    private int[] arr;
    private int size;

    // ------------------------------ Constructors

    /**
     * 优先队列数组默认大小为64
     */
    public PriorityHeap() {
        this(64);
    }

    public PriorityHeap(int initSize) {
        if (initSize <= 0) {
            initSize = 64;
        }
        this.arr = new int[initSize];
        this.size = 0;
    }

    // ------------------------------ Public methods

    public int max() {
        return this.arr[0];
    }

    public int maxAndRemove() {
        int t = max();

        this.arr[0] = this.arr[--size];
        sink(0, this.arr[0]);
        return t;
    }
    public void add(int data) {
        resize(1);
        this.arr[size++] = data;
        pop(size - 1, data);
    }

    // ------------------------------ Private methods

    /**
     * key下沉方法
     */
    private void sink(int i, int key) {
        while (2 * i <= this.size - 1) {
            int child = 2 * i;
            if (child < this.size - 1 && this.arr[child] < this.arr[child + 1]) {
                child++;
            }
            if (this.arr[i] >= this.arr[child]) {
                break;
            }

            swap(i, child);
            i = child;
        }
    }

    /**
     * key上浮方法
     */
    private void pop(int i, int key) {
        while (i > 0) {
            int parent = i / 2;
            if (this.arr[i] <= this.arr[parent]) {
                break;
            }
            swap(i, parent);
            i = parent;
        }
    }

    /**
     * 重新调整数组大小
     */
    private void resize(int increaseSize) {
        if ((this.size + increaseSize) > this.arr.length) {
            int newSize = (this.size + increaseSize) > 2 * this.arr.length ? (this.size + increaseSize) : 2 * this.arr.length;
            int[] t = this.arr;

            this.arr = Arrays.copyOf(t, newSize);
        }
    }

    /**
     * Swaps arr[a] with arr[b].
     */
    private void swap(int a, int b) {
        int t = this.arr[a];
        this.arr[a] = this.arr[b];
        this.arr[b] = t;
    }
}
```


#### 5.2.3 实现堆排序
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
#### 5.2.4 利用有限集队列合并K个有序数组
```java
class Solution {
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
}
```
#### 5.2.5 ：求一组动态数据集合的最大Top K
```java
/*
*不太明白这道题的具体意思，如果是需要求一组数据的Top K,
*那直接，最大堆排序，然后取TopK加入到一个list中返回，就行
*/
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
	
	public static List findKByHeap(int[] array, int k) {
	    buildHeap(array, k);
	    for (int i = k + 1; i < array.length; i++) {
	        if (array[i] > array[0]) {
	            int temp = array[i];
	            array[i] = array[0];
	            array[0] = temp;
	            maxHeapify(array, k, 0);
	        }
	    }
	    List list=new ArrayList();
	    for(int i =0;i<k;i++){
	    	list.add(array[i])
	    }
	    return list;
	}
```
#### 5.2.6 练习：路径总和
```java
class Solution {
    public boolean hasPathSum(TreeNode root, int sum) {
        if(root == null) return false;
        
        int t=sum-root.val;
        
       if(root.left==null && root.right==null)
           return t==0 ? true : false;
        
        return hasPathSum(root.left,t) || hasPathSum(root.right,t);
        
    }
}
```
