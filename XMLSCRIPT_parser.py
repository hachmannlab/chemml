from lxml import objectify, etree

def main(SCRIPT_NAME):
    doc = etree.parse(SCRIPT_NAME)
    cmls = etree.tostring(doc) 	# ChemML script : cmls
    #print cmls
    cmls = objectify.fromstring(cmls)
    objectify.deannotate(cmls)
    etree.cleanup_namespaces(cmls)
    print "\n"
    print(objectify.dump(cmls))
#     todo_order = [ element.tag for element in cmls.iterchildren() if element.status == 'on']
#     for element in cmls.iterchildren():
#         if element.attrib['status']=='on':
#             todo_order.append(element.tag)
#         elif element.attrib['status']=='sub':
#             for sub_element in element.iterchildren():
#                 if sub_element.attrib['status']=='on':
#                     todo_order.append(sub_element.tag)
    return cmls
                    

    
cmls = main('SCRIPT_NAME.xml')