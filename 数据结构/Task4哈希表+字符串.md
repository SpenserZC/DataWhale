# Datawhale 系列数据结构
## Task4.1  散列表
**基本概念**
	散列表(Hash  Table，又叫哈希表)，是根据关键码值(Key  Value)而直接进行访问的数据结构。也就是说，它通过把关键码值映射到表中一个位置来访问记录，以加快查找的速度。这个映射函数叫做散列函数，存放记录的数组叫做散列表。
**散列表思想**
	（1）使用散列函数将给定键转化为一个“数组的索引”，理想情况下，不同的key会被转化为不同的索引，但是在实际情况中，我们会遇到不同的键转化为相同索引的情况，这种情况叫做散列冲突/碰撞，后文中会详细讲解；
	（2）得到了索引后，我们就可以像访问数组一样，通过这个索引访问到相应的键值对。
**如何设计散列函数**
	（1）散列函数的设计不能太复杂：减少计算时间
	（2）散列函数整成的值要尽可能随机并且均匀分布
	主要方法有：
		a.直接寻址法
		b.数字分析法
		c.平方取中法
		d.折叠法
		e.随机数法
		f.除留取余法
**散列冲突**
	再好的散列函数也无法避免散列冲突
	主要方法有：
		直接寻址法
		链表法：更常用，4.1.1基于其设计散列表
#### 4.1.1实现一个基于链表解决冲突问题的散列表
```java
/*布谷鸟散列概述
	使用hashA、hashB计算对应的key位置：
    1、两个位置均为空，则任选一个插入； 
    2、两个位置中一个为空，则插入到空的那个位置 
    3、两个位置均不为空，则踢出一个位置后插入，被踢出的对调用该算法，再执行该算法找其另一个位置，循环直到插入成功。 
    4、如果被踢出的次数达到一定的阈值，则认为hash表已满，并进行重新哈希rehash
	cuckoo hashing的哈希函数是成对的（具体的实现可以根据需求设计），每一个元素都是两个，分别映射到两个位置，一个是记录的位置，另一个是备用位置。这个备用位置是处理碰撞时用的，cuckoo hashing处理碰撞的方法，就是把原来占用位置的这个元素踢走，不过被踢出去的元素还有一个备用位置可以安置，如果备用位置上还有人，再把它踢走，如此往复。直到被踢的次数达到一个上限，才确认哈希表已满，并执行rehash操作
*/
interface HashFamily<AnyType>{
	//根据which来选择散列函数，并返回hash值
	int hash(AnyType x,int which);
	//返回集合中散列的个数
	int getNumberOfFunctions();
	//获取新的散列函数
	void generateNewFunctions();
}

class CuckooHashTable<AnyType>{
	//定义最大装填因子为0.4
	  private static final double MAX_LOAD = 0.4;
	  //定义rehash次数达到一定时，进行再散列
	  private static final int ALLOWED_REHASHES = 1;
	  //定义默认表的大小
	  private static final int DEFAULT_TABLE_SIZE = 101;
	  //定义散列函数集合
	  private final HashFamily<? super AnyType> hashFunctions;
	  //定义散列函数个数
	  private final int numHashFunctions;
	  //定义当前表
	  private AnyType[] array;
	  //定义当前表的大小
	  private int currentSize;
	  //定义rehash的次数
	  private int rehashes = 0;
	  //定义一个随机数
	  private Random r = new Random();
	  
	  public CuckooHashTable(HashFamily<? super AnyType> hf){
	      this(hf, DEFAULT_TABLE_SIZE);
	  }
	  public void printArray() {
		// TODO Auto-generated method stub
		
	}
	//初始化操作
	  public CuckooHashTable(HashFamily<? super AnyType> hf, int size){
	      allocateArray(nextPrime(size));
	      doClear();
	      hashFunctions = hf;
	      numHashFunctions = hf.getNumberOfFunctions();
	  }

	  private int nextPrime(int size) {
		return size*2;
	}
	public void makeEmpty(){
	      doClear();
	  }
	  //清空操作
	  private void doClear(){
	      currentSize = 0;
	      for (int i = 0; i < array.length; i ++){
	          array[i] = null;
	      }
	  }
	  //初始化表
	  @SuppressWarnings("unchecked")
	private void allocateArray(int arraySize){
	      array = (AnyType[]) new Object[arraySize];
	  }
	  /**
	   *
	   * @param x 当前的元素
	   * @param which 选取的散列函数对应的位置
	   * @return
	   */
	  private int myHash(AnyType x, int which){
	      //调用散列函数集合中的hash方法获取到hash值
	      int hashVal = hashFunctions.hash(x, which);
	      //再做一定的处理
	      hashVal %= array.length;
	      if (hashVal < 0){
	          hashVal += array.length;
	      }
	      return hashVal;
	  }
	  /**
	   * 查询元素的位置，若找到元素，则返回其当前位置，否则返回-1
	   * @param x
	   * @return
	   */
	  private int findPos(AnyType x){
	      //遍历散列函数集合，因为不确定元素所用的散列函数为哪个
	      for (int i = 0; i < numHashFunctions; i ++){
	          //获取到当前hash值
	          int pos = myHash(x, i);
	          //判断表中是否存在当前元素
	          if (array[pos] != null && array[pos].equals(x)){
	              return pos;
	          }
	      }
	      return -1;
	  }
	public boolean contains(AnyType x){
	      return findPos(x) != -1;
	  }
	/**
	   * 删除元素：先查询表中是否存在该元素，若存在，则进行删除该元素
	   * @param x
	   * @return
	   */
	  public boolean remove(AnyType x){
	      int pos = findPos(x);
	      if (pos != -1){
	          array[pos] = null;
	          currentSize --;
	      }
	      return pos != -1;
	  }

	/**
	   * 插入：先判断该元素是否存在，若存在，在判断表的大小是否达到最大负载，
	   * 若达到，则进行扩展，最后调用insertHelper方法进行插入元素
	   * @param x
	   * @return
	   */
	  public boolean insert(AnyType x){
	      if (contains(x)){
	          return false;
	      }
	      if (currentSize >= array.length * MAX_LOAD){
	          expand();
	      }
	      return insertHelper(x);
	  }

	private boolean insertHelper(AnyType x) {
	        //记录循环的最大次数
	        final int COUNT_LIMIT = 100;
	        while (true){
	            //记录上一个元素位置
	            int lastPos = -1;
	            int pos;
	            //进行查找插入
	            for (int count = 0; count < COUNT_LIMIT; count ++){
	                for (int i = 0; i < numHashFunctions; i ++){
	                    pos = myHash(x, i);
	                    //查找成功，直接返回
	                    if (array[pos] == null){
	                        array[pos] = x;
	                        currentSize ++;
	                        return true;
	                    }
	                }
	                //查找失败，进行替换操作，产生随机数位置，当产生的位置不能与原来的位置相同
	                int i = 0;
	                do {
	                    pos = myHash(x, r.nextInt(numHashFunctions));
	                } while (pos == lastPos && i ++ < 5);
	                //进行替换操作
	                AnyType temp = array[lastPos = pos];
	                array[pos] = x;
	                x = temp;
	            }
	            //超过次数，还是插入失败，则进行扩表或rehash操作
	            if (++ rehashes > ALLOWED_REHASHES){
	                expand();
	                rehashes = 0;
	            } else {
	                rehash();
	            }
	        }
	    }

	private void expand(){
	        rehash((int) (array.length / MAX_LOAD));
	    }

	    private void rehash(){
	        hashFunctions.generateNewFunctions();
	        rehash(array.length);
	    }

	    private void rehash(int newLength){
	        AnyType [] oldArray = array;
	        allocateArray(nextPrime(newLength));
	        currentSize = 0;
	        for (AnyType str : oldArray){
	            if (str != null){
	                insert(str);
	            }
	        }
	    }

}
```
#### 4.1.2 实现一个LRU缓存淘汰算法
```java
class LRULinkedHashMap<K, V> extends LinkedHashMap<K, V> {  
    /**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private final int maxCapacity;  
   
    private static final float DEFAULT_LOAD_FACTOR = 0.75f;  
   
    private final Lock lock = new ReentrantLock();  
   
    public LRULinkedHashMap(int maxCapacity) {  
        super(maxCapacity, DEFAULT_LOAD_FACTOR, true);  
        this.maxCapacity = maxCapacity;  
    }  
   
    @Override 
    protected boolean removeEldestEntry(java.util.Map.Entry<K, V> eldest) {  
        return size() > maxCapacity;  
    }  
    @Override 
    public boolean containsKey(Object key) {  
        try {  
            lock.lock();  
            return super.containsKey(key);  
        } finally {  
            lock.unlock();  
        }  
    }  
   
       
    @Override 
    public V get(Object key) {  
        try {  
            lock.lock();  
            return super.get(key);  
        } finally {  
            lock.unlock();  
        }  
    }  
   
    @Override 
    public V put(K key, V value) {  
        try {  
            lock.lock();  
            return super.put(key, value);  
        } finally {  
            lock.unlock();  
        }  
    }  
   
    public int size() {  
        try {  
            lock.lock();  
            return super.size();  
        } finally {  
            lock.unlock();  
        }  
    }  
   
    public void clear() {  
        try {  
            lock.lock();  
            super.clear();  
        } finally {  
            lock.unlock();  
        }  
    }  
   
    public Collection<Map.Entry<K, V>> getAll() {  
        try {  
            lock.lock();  
            return new ArrayList<Map.Entry<K, V>>(super.entrySet());  
        } finally {  
            lock.unlock();  
        }  
    }  
}


```
#### 4.1.3 练习：两数之和
```java
class Solution {
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
}
```

## TASK4.2 字符串
#### 4.2.1 实现一个字符集，只包含这26个英文字母的Trie树
```java
class Trie_Tree{
	 
	private class Node{
		private int dumpli_num;////该字串的反复数目，  该属性统计反复次数的时候实用,取值为0、1、2、3、4、5……
		private int prefix_num;///以该字串为前缀的字串数。 应该包含该字串本身。。！
		private Node childs[];////此处用数组实现，当然也能够map或list实现以节省空间
		private boolean isLeaf;///是否为单词节点
		public Node(){
			dumpli_num=0;
			prefix_num=0;
			isLeaf=false;
			childs=new Node[26];
		}
	}	
	
	
	private Node root;///树根  
	public Trie_Tree(){
		///初始化trie 树
		root=new Node();
	}
	
	
	
	/**
	 * 插入字串。用循环取代迭代实现
	 * @param words
	 */
	public void insert(String words){
		insert(this.root, words);
	}
	/**
	 * 插入字串，用循环取代迭代实现
	 * @param root
	 * @param words
	 */
	private void insert(Node root,String words){
		words=words.toLowerCase();////转化为小写
		char[] chrs=words.toCharArray();
		
		for(int i=0,length=chrs.length; i<length; i++){
			///用相对于a字母的值作为下标索引，也隐式地记录了该字母的值
			int index=chrs[i]-'a';
			if(root.childs[index]!=null){
				////已经存在了，该子节点prefix_num++
				root.childs[index].prefix_num++;
			}else{
				///假设不存在
				root.childs[index]=new Node();
				root.childs[index].prefix_num++;				
			}	
			
			///假设到了字串结尾，则做标记
			if(i==length-1){
				root.childs[index].isLeaf=true;
				root.childs[index].dumpli_num++;
			}
			///root指向子节点，继续处理
			root=root.childs[index];
		}
		
	}
	
	/**
	 * 遍历Trie树，查找全部的words以及出现次数
	 * @return HashMap<String, Integer> map
	 */
	public HashMap<String,Integer> getAllWords(){
//		HashMap<String, Integer> map=new HashMap<String, Integer>();
			
		return preTraversal(this.root, "");
	}
	
	/**
	 * 前序遍历。。。
	 * @param root		子树根节点
	 * @param prefixs	查询到该节点前所遍历过的前缀
	 * @return
	 */
	private  HashMap<String,Integer> preTraversal(Node root,String prefixs){
		HashMap<String, Integer> map=new HashMap<String, Integer>();
		
		if(root!=null){
			
			if(root.isLeaf==true){
			////当前即为一个单词
				map.put(prefixs, root.dumpli_num);
			}
			
			for(int i=0,length=root.childs.length; i<length;i++){
				if(root.childs[i]!=null){
					char ch=(char) (i+'a');
					////递归调用前序遍历
					String tempStr=prefixs+ch;
					map.putAll(preTraversal(root.childs[i], tempStr));
				}
			}
		}		
		
		return map;
	}

	/**
	 * 推断某字串是否在字典树中
	 * @param word
	 * @return true if exists ,otherwise  false 
	 */
	public boolean isExist(String word){
		return search(this.root, word);
	}
	/**
	 * 查询某字串是否在字典树中
	 * @param word
	 * @return true if exists ,otherwise  false 
	 */
	private boolean search(Node root,String word){
		char[] chs=word.toLowerCase().toCharArray();
		for(int i=0,length=chs.length; i<length;i++){
			int index=chs[i]-'a';
			if(root.childs[index]==null){
				///假设不存在，则查找失败
				return false;
			}			
			root=root.childs[index];			
		}
		
		return true;
	}
	
	/**
	 * 得到以某字串为前缀的字串集。包含字串本身。 相似单词输入法的联想功能
	 * @param prefix 字串前缀
	 * @return 字串集以及出现次数，假设不存在则返回null
	 */
	public HashMap<String, Integer> getWordsForPrefix(String prefix){
		return getWordsForPrefix(this.root, prefix);
	}
	/**
	 * 得到以某字串为前缀的字串集。包含字串本身。
	 * @param root
	 * @param prefix
	 * @return 字串集以及出现次数
	 */
	private HashMap<String, Integer> getWordsForPrefix(Node root,String prefix){
		HashMap<String, Integer> map=new HashMap<String, Integer>();
		char[] chrs=prefix.toLowerCase().toCharArray();
		////
		for(int i=0, length=chrs.length; i<length; i++){
			
			int index=chrs[i]-'a';
			if(root.childs[index]==null){
				return null;
			}
			
			root=root.childs[index];
		
		}
		///结果包含该前缀本身
		///此处利用之前的前序搜索方法进行搜索
		return preTraversal(root, prefix);
	}   
}
```

#### 4.2.2 实现朴素的字符串匹配算法
```java
public static int indext(String src, String target) {
        return indext(src,target,0);
    }

    public static int indext(String src, String target, int fromIndex) {
        return indext(src.toCharArray(), src.length(), target.toCharArray(), target.length(), fromIndex);
    }

    //朴素模式匹配算法
    static int indext(char[] s, int slen, char[] t, int tlen, int fromIndex) {
        if (fromIndex < 0) {
            fromIndex = 0;
        }
        if (tlen == 0) {
            return fromIndex;
        }
        if (slen == 0) {
            return -1;
        }
        int i = fromIndex;
        int j = 0;
        while (i <= slen && j <= tlen) {
            /*  cycle compare */
            if (s[i] == t[j]) {
                ++i;
                ++j;
            } else {
                /*  point back last position */
                i = i - j + 1;
                j = 0;
            }
        }
        if (j > tlen) {
            /*  found target string retun first index position*/
            return i - j;
        } else {
             /* can't find target  string and retun -1 */
            return -1;
        }
    }
```
#### 3.2.3 练习：反转字符串
```java
class Solution {
    public String reverseString(String s) {
      final char[] array = s.toCharArray();
      final int length = array.length;
      for (int i = 0; i < length / 2; i++) {
        char temp = array[i];
        array[i] = array[length - i-1];
        array[length - i-1] = temp;
      }
      return new String(array);
    }
}
```
#### 3.2.3 练习：反转字符串里的单词
```java
 public String reverseWords(String s) {
        String[] words = s.split(" ");
        StringBuilder sb = new StringBuilder();
        for(String word : words) {
            sb.append(swapWord(0, word.length()-1, word.toCharArray())).append(" ");
        }
        
        return sb.toString().trim();
    }
    
    public String swapWord(int s, int e, char[] c) {
        if(s >= e) {
            return String.valueOf(c);
        }
        
        char temp = c[s];
        c[s] = c[e];
        c[e] = temp;
        return swapWord(s+1, e-1, c);
    }
```
#### 3.2.3 练习：字符串转换整数(stoi)
```java
 public int myAtoi(String str) {
        //去除掉前后的空格
        String strr = str.trim();
        //存储最终过滤出来的字符串
        String strrr = null;
        //字符串不为空时并且字符串不全是空白字符串时才转换
        if(strr != null && strr.isEmpty() == false){
            char f = strr.charAt(0);
            //判断字符串中的第一个非空格字符是不是一个有效整数字符
            if(f >= '0' && f <= '9' || f == '+'|| f == '-'){
                strrr = strr.substring(0,1); // 把第一位放进去(只能是数字、正负号)
                //这时候循环只要数字，因为正负号只能出现在第一位
                for(int i = 1; i<strr.length();i++){
                    if(strr.charAt(i) >= '0' && strr.charAt(i) <= '9'){
                        strrr = strr.substring(0,i+1);
                    }
                    //这是遇到不符合要求的字符，直接忽略剩余元素
                    else{break;}
                }
            }
        }
        //判断最终字符串是否为空或则只有一个正负号
        if(strrr == null || strrr.equals("+") || strrr.equals("-"))
            //此时strrr是String对象，如果使用==比较则比较的时内存地址
            return 0;
        //最终转换成的数字
        int num = 0;
        //使用异常机制打印结果
        try{
            num = Integer.parseInt(strrr);
        }catch (Exception e){
            if(strrr.charAt(0) == '-')
                return Integer.MIN_VALUE;
            return Integer.MAX_VALUE;
        }
        return num;
    }
```
参考文章：
	散列表参考文章：
	https://blog.csdn.net/ynnusl/article/details/89343419
	https://blog.csdn.net/u012124438/article/details/78230478
	字符串参考文章：
	https://www.cnblogs.com/lcchuguo/p/5194323.html