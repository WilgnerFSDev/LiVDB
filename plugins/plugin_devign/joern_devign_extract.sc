import java.io.{BufferedWriter, FileWriter, File}

@main def exec(
    cpgPath:   String,
    outputDir: String,
    minNodes:  Int = 1,
    maxNodes:  Int = 500,
    minEdges:  Int = 1
) = {

  println(s"[joern_devign] Starting extraction. CPG: $cpgPath  Output: $outputDir")
  println(s"[joern_devign] Filters - minNodes=$minNodes  maxNodes=$maxNodes (Devign <= 500)  minEdges=$minEdges")

  new File(outputDir).mkdirs()
  importCpg(cpgPath)

  def escapeJson(s: String): String = {
    if (s == null) ""
    else s.replace("\\", "\\\\")
          .replace("\"", "\\\"")
          .replace("\n", "\\n")
          .replace("\r", "\\r")
          .replace("\t", "\\t")
          .replace("\b", "\\b")
          .replace("\f", "\\f")
  }

  def labelFromFilename(filename: String): Int = {
    val base  = new File(filename).getName.replaceAll("\\.(c|cpp|h|hpp)$", "")
    val parts = base.split("_")
    scala.util.Try(parts.last.toInt).getOrElse(0)
  }

  def nodeOrder(n: io.shiftleft.codepropertygraph.generated.nodes.AstNode): (Int, Int, Int) = {
    val line = scala.util.Try(n.lineNumber.map(_.intValue()).getOrElse(0)).getOrElse(0)
    val col  = scala.util.Try(n.columnNumber.map(_.intValue()).getOrElse(0)).getOrElse(0)
    val ord  = scala.util.Try(n.order.intValue()).orElse(scala.util.Try(n.order.toInt)).getOrElse(0)
    (line, col, ord)
  }

  implicit val orderTuple: Ordering[(Int, Int, Int)] = Ordering.Tuple3[Int, Int, Int]

  var written = 0
  var skipped = 0

  cpg.method
    .filter(m =>
        m.filename != "<empty>" &&
        (m.filename.endsWith(".c") || m.filename.endsWith(".cpp") || m.filename.endsWith(".h") || m.filename.endsWith(".hpp")) &&
        !m.name.startsWith("<")
    )
    .foreach { method =>

      val allAstNodes = method.ast.l
      val astCount    = allAstNodes.size

      if (astCount < minNodes || astCount > maxNodes) {
        skipped += 1
      } else {

        val methodName = method.name
        val filename   = method.filename
        val label      = labelFromFilename(filename)

        val nodeMap = scala.collection.mutable.LinkedHashMap[Long, (String, String)]()
        allAstNodes.foreach { n =>
          val code     = scala.util.Try(n.code).getOrElse("")
          val nodeType = n.label
          nodeMap(n.id) = (code, nodeType)
        }

        val domainIds = nodeMap.keySet
        val edgeBuf   = scala.collection.mutable.ArrayBuffer[(Long, String, Long)]()

        allAstNodes.foreach { src =>
          src.astChildren.foreach { dst =>
            if (domainIds.contains(dst.id) && src.id != dst.id) {
              edgeBuf += ((src.id, "AST", dst.id))
            }
          }
        }

        allAstNodes.foreach { src =>
          src.outE("CFG").foreach { edge =>
            val dstId = edge.dst.id
            if (domainIds.contains(dstId) && src.id != dstId) {
              edgeBuf += ((src.id, "CFG", dstId))
            }
          }
        }

        val leaves = allAstNodes
          .filter(n => n.astChildren.isEmpty && n.code != null && n.code.trim.nonEmpty)
          .sortBy(nodeOrder)

        for (i <- 0 until leaves.size - 1) {
          val srcLeaf = leaves(i)
          val dstLeaf = leaves(i + 1)
          if (srcLeaf.id != dstLeaf.id) {
            edgeBuf += ((srcLeaf.id, "NCS", dstLeaf.id))
          }
        }

        val assignments = allAstNodes.collect {
          case c: io.shiftleft.codepropertygraph.generated.nodes.Call
            if c.name.startsWith("<operator>.assignment") => c
        }

        val writeNodeIds = scala.collection.mutable.HashSet[Long]()

        assignments.foreach { call =>
          scala.util.Try {
            val lhs   = call.argument(1)
            val rhs   = call.argument(2)
            val lhsId = lhs.id
            writeNodeIds += lhsId

            rhs.ast.isIdentifier.foreach { rVar =>
              if (domainIds.contains(rVar.id) && domainIds.contains(lhsId) && rVar.id != lhsId) {
                edgeBuf += ((rVar.id, "DFG_C", lhsId))
              }
            }
          }
        }

        allAstNodes.collect {
          case p: io.shiftleft.codepropertygraph.generated.nodes.MethodParameterIn => p
        }.foreach { p =>
          writeNodeIds += p.id
        }

        val varsByName = allAstNodes
          .collect { case id: io.shiftleft.codepropertygraph.generated.nodes.Identifier => id }
          .groupBy(_.name)

        varsByName.foreach { case (varName, occurrences) =>
          if (varName != null && varName.trim.nonEmpty) {
            val sortedOccurrences = occurrences.sortBy(nodeOrder)
            var lastReadId:  Option[Long] = None
            var lastWriteId: Option[Long] = None

            allAstNodes.collect {
              case p: io.shiftleft.codepropertygraph.generated.nodes.MethodParameterIn
                if p.name == varName => p
            }.headOption.foreach { p =>
              lastWriteId = Some(p.id)
            }

            sortedOccurrences.foreach { occ =>
              val occId   = occ.id
              val isWrite = writeNodeIds.contains(occId)

              if (isWrite) {

                if (lastWriteId.isDefined && lastWriteId.get != occId) {
                  edgeBuf += ((lastWriteId.get, "DFG_W", occId))
                }
                lastWriteId = Some(occId)
              } else {

                if (lastReadId.isDefined && lastReadId.get != occId) {
                  edgeBuf += ((lastReadId.get, "DFG_R", occId))
                }
                lastReadId = Some(occId)

                if (lastWriteId.isDefined && lastWriteId.get != occId) {
                  edgeBuf += ((lastWriteId.get, "DFG_W", occId))
                }
              }
            }
          }
        }

        val numEdges = edgeBuf.size

        if (numEdges < minEdges) {
          skipped += 1
        } else {

          val idList    = nodeMap.keys.toSeq
          val idToIndex = idList.zipWithIndex.toMap

          val nodeSb = new java.lang.StringBuilder(nodeMap.size * 96)
          idList.zipWithIndex.foreach { case (id, i) =>
            if (i > 0) nodeSb.append(',')
            val (code, ntype) = nodeMap(id)
            nodeSb.append(s"""{"id":$i,"code":"${escapeJson(code)}","node_type":"${escapeJson(ntype)}"}""")
          }

          val edgeSb = new java.lang.StringBuilder(edgeBuf.size * 32)
          var firstEdge = true
          edgeBuf.foreach { case (src, etype, dst) =>
            if (idToIndex.contains(src) && idToIndex.contains(dst)) {
              if (!firstEdge) edgeSb.append(',')
              edgeSb.append(s"""[${idToIndex(src)},"${escapeJson(etype)}",${idToIndex(dst)}]""")
              firstEdge = false
            }
          }

          val safeMethod = methodName.replaceAll("[^a-zA-Z0-9_\\-]", "_")
          val safeFile   = new File(filename).getName.replaceAll("\\.(c|cpp|h|hpp)$", "")
          val outPath    = s"$outputDir/${safeFile}__${safeMethod}.json"

          val json =
            s"""{"func_name":"${escapeJson(methodName)}","label":$label,"nodes":[${nodeSb}],"edges":[${edgeSb}]}"""

          val bw = new BufferedWriter(new FileWriter(outPath), 65536)
          try { bw.write(json) } finally { bw.close() }

          written += 1
        }
      }
    }

  println(s"[joern_devign] Done. Written=$written  Skipped(filters)=$skipped  Output: $outputDir")
}
