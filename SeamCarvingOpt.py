'''
SeamCarving.py
The program implements vertical seam carving with dynamic programming methods.
The default setting is to remove 99 pixels as the project guidelines indicate.
Group members: Seunghyeon (Hyeon) Kim, Joyce Gill
'''

# Import numpy for quicker array operations
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# class SeamCarving()
# This is a class that utilizes various functions to SeamCarve given images.
class SeamCarving():
    '''
    This class seam carves the given image with its iterators returning the numpy array of 
    the resultant image.
    Author(s): Seunghyeon (Hyeon) Kim, Joyce Gill
    '''
    def __init__(self, imageDir : str, maximumIter: int):
        '''
        Image Directory and the maximum number of iterations (number of columns to be carved)
        '''
        self.pixArr = np.asarray(Image.open(imageDir))
        self.dim = self.pixArr.shape
        self.maxIt = maximumIter
        # mod == 0 for vertical (default).
        # mod == 1 for horizontal.

    def __iter__(self):
        '''
        Iterator for the seam carving object. It initializes the iterator.
        '''
        self.ind = 0
        return self
    
    def __next__(self):
        '''
        Next object. In this case, it will be the object with one seam removed.
        '''
        self.calcEnergyArr()
        self.optimalVerticalPath()
        if(self.ind == 0):
            np.savetxt('energy.csv', self.energyArr[1:-1, 1:-1], fmt='%.3f', delimiter = ',')
            (self.optP[:,1]).tofile('seam1.csv', sep = ',')
        #self.redColorOptimal()
        self.pixArr = self.removeOptimal()
        self.dim = self.pixArr.shape
        self.ind += 1
        if (self.ind >= self.maxIt):
            img = Image.fromarray(self.pixArr.astype('uint8'), 'RGB')
            img.save('FinalImage.jpg')
            #img.show()
            raise StopIteration
        return self.pixArr
    
    def showImg(self):
        '''
        The function shows the image that has been carved.
        Comments: FOR DEBUGGING PURPOSES ONLY
        '''
        plt.imshow(self.pixArr, interpolation='nearest')
        plt.show()

    def calcEnergyArr(self):
        '''
        Pre Conditions: pixArr is the numpy pixel arrays with all the pixel values of the 
        input image.
        Post Conditions: The function returns the numpy array that ran through the energy 
        function on each of the pixels.
        '''
        # Define shifted arrays
        pixXP1 = np.roll(self.pixArr, (1, 0), axis = (0, 1)) # x axis shifted right side once
        pixXM1 = np.roll(self.pixArr, (-1, 0), axis = (0, 1)) # x axis shifted left side once
        pixYP1 = np.roll(self.pixArr, (1, 0), axis = (1, 0)) # y axis shifted right side once
        pixYM1 = np.roll(self.pixArr, (-1, 0), axis = (1, 0)) # y axis shifted left side once
        # Define the residual arrays for calculation
        residArrX = pixXP1-pixXM1 # residual array on x axis
        residArrY = pixYP1-pixYM1 # residual array on y axis
        # Define the sum of the residuals squared.
        sumOfResidX = np.sum(np.square(residArrX), axis = 2)
        sumOfResidY = np.sum(np.square(residArrY), axis = 2)
        # Define the sum of the residuals squared combined with x axis and y axis and then square rooted.
        sumOfResid = np.sqrt(sumOfResidX+sumOfResidY)
        # Replace the values at the edges with 1000.
        sumOfResid[0,:] = 1000
        sumOfResid[:,0] = 1000
        sumOfResid[-1,:] = 1000
        sumOfResid[:,-1] = 1000
        # Add energy array
        self.energyArr = sumOfResid
        return self.energyArr
    
    #def findSeam(self, current):

    def optimalVerticalPath(self):
        '''
        Pre Conditions: Proper initialization of the class and calcEnergyArr must be done 
        before.
        Post Conditions: This function will store the optimal vertical path (least 
        energetic path) as a numpy array.
        '''
        totalPath = np.array([])
        totalEnergy = np.array([])
        optimalPath = []
        optimalPathEnergy = 0
        for x in range(0, self.dim[1]):
            # Declare temporary index variables
            current = (0, x)
            # Record which path was passed by
            passedPt = []
            passedEnergy = []
            cumulatedEnergy = 0
            # As long as the indices are not out of bounds, run the loop
            while (current[0] < self.dim[0]-1 and current[1] < self.dim[1]):
                # Lowest Upper Bound of the vertices
                supremum = min(current[1]+1, self.dim[1]-1)
                # Greatest Lower Bound of the vertices
                infimum = max(current[1]-1, 0)
                # Use the range to extract the indices
                ran = range(infimum, supremum+1)
                nextRow = current[0]+1
                candidates = [(nextRow, k) for k in ran]
                candidateEnergy = np.array([self.energyArr[nextRow,k] for k in ran])
                # Next point will be the minimum energy point out of the candidates
                nextPtInd = candidates[np.argmin(candidateEnergy)]
                # When we know what the next candidate is, add the current point's energy to the accumulated variable
                cumulatedEnergy += self.energyArr[current]
                # When the point is passed, append it to the passed points array
                passedPt.append(current)
                passedEnergy.append(cumulatedEnergy)
                # Then go to the next index
                current = nextPtInd
                if(totalPath.size != 0):
                    someInd = np.where(totalPath[:, current[0]] == current)[0][0]
                    if(someInd != -1):
                        passedPt.extend(totalPath[someInd][current[0]:-1])
                        passedEnergy.extend(totalEnergy[someInd][current[0]:-1] - passedEnergy[-1])
                        break
            # After the loop, the border is not added due to the while loop conditions, so add it
            passedPt.append((self.dim[0]-1, passedPt[-1][1]))
            passedEnergy.append(passedEnergy[-1]+1000)
            if (totalPath.size == 0):
                totalPath = np.array([passedPt])
                totalEnergy = np.array([passedEnergy])
            else:
                np.append(totalPath, [passedPt], axis = 0)
                np.append(totalEnergy, [passedEnergy], axis=0)
            if((optimalPathEnergy >= passedEnergy[-1]) or (len(optimalPath) == 0)):
                optimalPath = passedPt
                optimalPathEnergy = passedEnergy[-1]
        # Select the seam with minimal accumulated energy
        self.optP = np.array(optimalPath)

    def redColorOptimal(self):
        '''
        Pre-condition: The optimal path must be calculated either vertical or horizontal.
        Post-condition: It will return the array with red-colored on the optimal path.
        Comments: FOR DEBUGGING USAGE ONLY.
        '''
        # Create a copy of the original picture
        copyArr = self.pixArr.copy()
        # Change the type from the original numpy array to list so that it is mutable
        copyList = list(copyArr.tolist())
        # Change the coordinates of the path to red color
        for coord in self.optP:
            copyList[coord[0]][coord[1]] = [255, 0, 0]
        # Repaste the list as a numpy array to copyArr
        copyArr = np.array(copyList)
        # Show the resultant image
        plt.imshow(copyArr, interpolation='nearest')
        plt.show()
    
    def removeOptimal(self):
        '''
        Pre-condition: The optimal path must be calculated either vertical or horizontal.
        Post-condition: It will return the array with pixels on the optimal path removed.
        '''
        copyArr = self.pixArr.copy()
        copyList = list(copyArr.tolist())
        for coord in self.optP:
            copyList[coord[0]].pop(coord[1])
        copyArr = np.array(copyList)
        return copyArr
    
def main():
    sc = SeamCarving("InitialImage-1.jpg", 99)
    iterSc = iter(sc)
    for _ in iterSc:
        print(sc.ind)
        _
    # for i in range(300):
    #     sc.calcEnergyArr()
    #     sc.optimalVerticalPath()
    #     temp = sc.removeOptimal()
    #     print(i)
    #plt.imshow(temp, interpolation='nearest')
    #plt.show()

main()