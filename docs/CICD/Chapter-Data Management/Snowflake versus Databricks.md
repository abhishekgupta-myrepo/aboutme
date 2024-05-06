# Snowflake Vs. Databricks

Created: 2022-06-28 09:47:11 -0400

Modified: 2022-06-29 07:54:58 -0400

---

![DATABRICKS ](snowflake_databricks_1.jpg)




<table>
<colgroup>
<col style="width: 22%" />
<col style="width: 39%" />
<col style="width: 37%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>Feature</strong></th>
<th><strong>Databricks</strong></th>
<th><strong>Snowflake</strong></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><strong>Founders</strong></td>
<td><ul class="incremental">
<li><p>Ex-Berkley, Authors of Spark. Donated Spark to Apache and then opened managed services company as Databricks.</p></li>
<li><p>Planned for IPO in 2022</p></li>
</ul></td>
<td><ul class="incremental">
<li><p>Ex-Oracle, Oracle principles behind the snowflake architecture</p></li>
<li><p>Public listed. Revenue $1.2B</p></li>
</ul></td>
</tr>
<tr class="even">
<td><strong>Architecture</strong></td>
<td>Big data distributed computing provided spark as a managed service.</td>
<td>SaaS Datawarehouse platform. Truly separates storage from compute.</td>
</tr>
<tr class="odd">
<td><strong>Implementation Cloud</strong></td>
<td>Tight with Azure but also available on GCP, AWS</td>
<td>Tight with AWS but also available on GCP, Azure</td>
</tr>
<tr class="even">
<td><strong>Machine Learning</strong></td>
<td><ul class="incremental">
<li><p>ML Lib library. Native to the Spark distribution.</p></li>
<li><p>ML Core is the core library that does internal management, scheduling, memory management.</p></li>
<li><p>Jupyter Notebook style environment to perform the ML experiments.</p></li>
</ul>
<blockquote>
<p></p>
</blockquote></td>
<td><ul class="incremental">
<li><p>Provides machine learning capability through newly launched feature Snowpark.</p></li>
<li><p>It is Snowflake + Spark providing connectors through java, scala, python api.</p></li>
<li><p>It requires the data to be available in the snowflake. Else you need to load the data from lake into snowflake.</p></li>
<li><p>Partner ecosystem that integrates with snowflake.</p></li>
</ul></td>
</tr>
<tr class="odd">
<td><strong>Data Ingestion</strong></td>
<td><ul class="incremental">
<li><p>ML Streams. Part of the Spark core distribution.</p></li>
<li><p>Provides both real-time events and batch ELT processing.</p></li>
<li><p>Loads the data from various data sources including data lake.</p></li>
</ul></td>
<td><ul class="incremental">
<li><p>Uses partner network to provide the ingestion.</p></li>
<li><p>Databricks is also a partner of snowflake</p></li>
<li><p>Has developed snowpipe for short batch mode data ingestion</p></li>
</ul></td>
</tr>
<tr class="even">
<td><strong>Use cases</strong></td>
<td><ul class="incremental">
<li><p>Focused on Machine learning use cases</p></li>
<li><p>Big data processing</p></li>
</ul></td>
<td><ul class="incremental">
<li><p>Focused on data warehousing use cases</p></li>
<li><p>Developing dashboards on top of processed data from snowflake</p></li>
</ul></td>
</tr>
<tr class="odd">
<td><strong>Pricing</strong></td>
<td><ul class="incremental">
<li><p>Pay as you go pricing. You pay for the databricks and the underlying resources such as the compute and storage used in the cloud</p></li>
</ul></td>
<td><ul class="incremental">
<li><p>Pay as you go. You pay for the storage of the data and the compute. This is in form a warehouse servers which can be spin up to perform the compute.</p></li>
</ul></td>
</tr>
<tr class="even">
<td><strong>Scalability</strong></td>
<td><ul class="incremental">
<li><p>Clusters can be scaled with 100's of spark executor nodes added to the cluster to perform the compute.</p></li>
<li><p>Clusters can be spin on for each job to be executed and then shut down the job completes.</p></li>
<li><p>Cluster can also be spin up for performing ML development</p></li>
<li><p>Has auto scaling mode.</p></li>
</ul></td>
<td><ul class="incremental">
<li><p>Warehouse machines are small, medium, large, X Large. Each has defined number of nodes that can be increased or decreased in auto-scaling mode.</p></li>
</ul></td>
</tr>
<tr class="odd">
<td><strong>Users</strong></td>
<td><ul class="incremental">
<li><p>Data Engineers, Data Scientist, Advanced Data analysts</p></li>
</ul></td>
<td><ul class="incremental">
<li><p>Data analysts, business users, Data engineers</p></li>
</ul></td>
</tr>
<tr class="even">
<td><strong>Coding languages</strong></td>
<td><ul class="incremental">
<li><p>Java, Scala (Major), Pyspark(Major), C, Hive SQL</p></li>
</ul></td>
<td><ul class="incremental">
<li><p>ANSI SQL, Python (Snowpark. New feature), Scala (Snowpark. New feature)</p></li>
</ul></td>
</tr>
</tbody>
</table>

